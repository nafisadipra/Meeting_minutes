# speaker_bank.py
import numpy as np
from scipy.spatial.distance import cosine


class SmartSpeakerBank:
    """
    Speaker bank with live-enrollment-first design.
    
    Critical change: pre-enrolled profiles are used only as HINTS to
    assign human-readable names to speakers discovered live, not as
    direct distance-matching anchors. This is because pyannote embeddings
    extracted under different recording conditions (enrollment mic vs
    meeting mic) often have cosine distances above 0.7 even for the
    same speaker, which causes false negatives.
    
    The matching pipeline is now:
    1. Live profiles (built from in-meeting audio) are the source of truth
       for distance matching.
    2. When a new live profile is created, we check if any pre-enrolled
       profile is its closest match and inherit that name as a hint.
    3. Otherwise, the speaker gets a Guest_NN name that you can rename
       after the meeting.
    """

    def __init__(
        self,
        anchor_profiles=None,
        expected_attendees=None,
        threshold=0.65,
        margin=0.05,
        new_speaker_threshold=0.72,
        use_anchors_as_hints_only=False,
    ):
        # Live profiles (built during meeting) - these drive matching
        self.centroids = {}
        # Total accumulated samples per centroid (used for adaptive alpha)
        self.centroid_sample_counts = {}
        # Number of update events per centroid
        self.centroid_counts = {}

        # Anchor profiles (pre-enrolled) - used for naming hints only
        self.anchor_profiles = {}
        if anchor_profiles:
            for name, vec in anchor_profiles.items():
                self.anchor_profiles[name] = np.array(vec).flatten()

        clean_attendees = expected_attendees.copy() if expected_attendees else []
        self.expected_attendees = [
            name for name in clean_attendees if name not in self.anchor_profiles
        ]

        self.threshold = threshold
        self.new_speaker_threshold = new_speaker_threshold
        self.margin = margin
        self.use_anchors_as_hints_only = use_anchors_as_hints_only
        self.next_guest_id = 1

        # If we are NOT using anchors as hints only, copy them as live centroids
        # (legacy behavior). Default for this version is hints-only.
        if not use_anchors_as_hints_only:
            for name, vec in self.anchor_profiles.items():
                self.centroids[name] = vec.copy()
                self.centroid_counts[name] = 1
                self.centroid_sample_counts[name] = 48000

    def _claim_name_from_anchors(self, vector):
        """
        Look at pre-enrolled anchor profiles. If one is closest to this
        new live profile AND is closer than any other anchor by a margin,
        claim that name. Otherwise return None.
        """
        if not self.anchor_profiles:
            return None

        vec_flat = np.asarray(vector).flatten()
        scored = []
        for name, anchor in self.anchor_profiles.items():
            dist = cosine(vec_flat, anchor)
            scored.append((name, dist))
        scored.sort(key=lambda x: x[1])

        best_name, best_dist = scored[0]

        # Loose threshold here: we're using this only as a naming hint,
        # not as a hard match. Even noisy cross-condition matches usually
        # land below 0.85 for the correct speaker.
        if best_dist > 0.85:
            return None

        # Require a margin over the second-best to avoid ambiguous claims
        if len(scored) > 1:
            second_dist = scored[1][1]
            if second_dist - best_dist < 0.05:
                return None

        # Remove the claimed anchor so it can't be claimed twice
        claimed_name = best_name
        del self.anchor_profiles[claimed_name]
        # Also remove from expected_attendees if present
        if claimed_name in self.expected_attendees:
            self.expected_attendees.remove(claimed_name)
        return claimed_name

    def _get_speaker_name(self, vector=None):
        """
        Generate a name for a newly-discovered speaker.
        Tries in order: anchor profile match, expected attendee, guest ID.
        """
        if vector is not None:
            claimed = self._claim_name_from_anchors(vector)
            if claimed:
                return claimed

        if self.expected_attendees:
            return self.expected_attendees.pop(0)

        name = f"Guest_{self.next_guest_id:02d}"
        self.next_guest_id += 1
        return name

    def _all_distances(self, vec_flat):
        distances = []
        for label, centroid in self.centroids.items():
            dist = cosine(vec_flat, centroid)
            distances.append((label, dist))
        distances.sort(key=lambda x: x[1])
        return distances

    def update_centroid(self, label, vector, n_samples=48000):
        """
        Update a centroid with a new sample, weighted by sample count.
        Larger samples get more weight; longer-established centroids
        update more slowly.
        """
        vec_flat = np.asarray(vector).flatten()
        if label not in self.centroids:
            return

        old = self.centroids[label]
        old_sample_count = self.centroid_sample_counts.get(label, 48000)
        new_total = old_sample_count + n_samples

        # Weighted average: each centroid is a weighted mean of all samples
        weight = n_samples / new_total
        # Cap weight to prevent any single sample from dominating
        weight = min(weight, 0.4)

        self.centroids[label] = (old * (1 - weight)) + (vec_flat * weight)
        self.centroid_sample_counts[label] = min(new_total, 240000)  # cap at 15s
        self.centroid_counts[label] = self.centroid_counts.get(label, 0) + 1

    def score_against_known(self, vector, target_label):
        vec_flat = np.asarray(vector).flatten()
        if target_label not in self.centroids:
            return self.process_segment(vector, allow_new_speaker=False, dry_run=True)

        target_dist = cosine(vec_flat, self.centroids[target_label])
        target_confidence = max(0.0, 1.0 - target_dist)

        distances = self._all_distances(vec_flat)
        best_label, best_dist = distances[0]

        if best_label == target_label:
            return target_label, target_confidence
        if target_dist - best_dist < self.margin:
            return target_label, target_confidence
        return best_label, max(0.0, 1.0 - best_dist)

    def process_segment(self, vector, allow_new_speaker=True, dry_run=False,
                        prefer_label=None, n_samples=48000):
        vec_flat = np.asarray(vector).flatten()

        # Cold start: no live centroids yet. Create the first one and
        # try to claim a name from anchor profiles.
        if not self.centroids:
            label = self._get_speaker_name(vector=vec_flat)
            if not dry_run:
                self.centroids[label] = vec_flat
                self.centroid_counts[label] = 1
                self.centroid_sample_counts[label] = n_samples
            return label, 1.0

        distances = self._all_distances(vec_flat)
        best_label, best_dist = distances[0]

        # Continuity preference
        if prefer_label and prefer_label in self.centroids:
            prefer_dist = cosine(vec_flat, self.centroids[prefer_label])
            if prefer_dist - best_dist < self.margin * 2:
                best_label = prefer_label
                best_dist = prefer_dist

        confidence = max(0.0, 1.0 - best_dist)

        if len(distances) > 1:
            second_dist = distances[1][1]
            ambiguity = second_dist - best_dist
            if ambiguity < self.margin:
                confidence *= 0.7

        if best_dist < self.threshold:
            if not dry_run:
                self.update_centroid(best_label, vec_flat, n_samples=n_samples)
            return best_label, confidence

        # Uncertain zone: still likely the same speaker, just with variation
        if best_dist < self.new_speaker_threshold:
            if not dry_run:
                self.update_centroid(best_label, vec_flat, n_samples=n_samples)
            return best_label, confidence * 0.8

        if not allow_new_speaker:
            return best_label, confidence * 0.5

        # New speaker: try to claim an anchor name
        label = self._get_speaker_name(vector=vec_flat)
        if not dry_run:
            self.centroids[label] = vec_flat
            self.centroid_counts[label] = 1
            self.centroid_sample_counts[label] = n_samples
        return label, 1.0

    def get_finalized_centroids(self):
        """
        Return all live centroids. After the meeting, you can also try
        to retroactively assign anchor names to any unclaimed Guest_NN
        speakers by checking distances against remaining anchors.
        """
        result = dict(self.centroids)

        # Try to retroactively rename Guest_NN speakers to anchor names
        if self.anchor_profiles:
            guests = [n for n in result if n.startswith("Guest_")]
            for guest in guests:
                guest_vec = result[guest]
                best_anchor = None
                best_dist = float('inf')
                for anchor_name, anchor_vec in self.anchor_profiles.items():
                    dist = cosine(guest_vec, anchor_vec)
                    if dist < best_dist:
                        best_dist = dist
                        best_anchor = anchor_name
                if best_anchor and best_dist < 0.85:
                    result[best_anchor] = result.pop(guest)
                    del self.anchor_profiles[best_anchor]

        return result