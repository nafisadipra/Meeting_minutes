import numpy as np
from scipy.spatial.distance import cosine


class SmartSpeakerBank:
    """
    Live-enrollment-first speaker bank.

    Pre-enrolled anchor profiles name speakers but do not drive matching.
    Cross-condition ECAPA embeddings (enrollment mic vs. meeting mic) are
    too variable for reliable direct matching, so live centroids built from
    in-meeting audio own all distance comparisons. Anchors are consumed
    once to claim a name for a newly created live centroid; unclaimed
    anchors get a second chance during finalization.
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
        self.centroids: dict[str, np.ndarray] = {}
        self.centroid_sample_counts: dict[str, int] = {}
        self.centroid_counts: dict[str, int] = {}

        self.anchor_profiles: dict[str, np.ndarray] = {}
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

        if not use_anchors_as_hints_only:
            for name, vec in self.anchor_profiles.items():
                self.centroids[name] = vec.copy()
                self.centroid_counts[name] = 1
                self.centroid_sample_counts[name] = 48000

    def _claim_name_from_anchors(self, vector: np.ndarray) -> str | None:
        """Consume the best-matching anchor name for a newly created live centroid."""
        if not self.anchor_profiles:
            return None

        vec_flat = np.asarray(vector).flatten()
        scored = []
        for name, anchor in self.anchor_profiles.items():
            dist = cosine(vec_flat, anchor)
            scored.append((name, dist))
        scored.sort(key=lambda x: x[1])

        best_name, best_dist = scored[0]

        # Cross-condition same-speaker distances rarely exceed 0.85 for ECAPA-TDNN.
        if best_dist > 0.85:
            return None

        if len(scored) > 1:
            second_dist = scored[1][1]
            if second_dist - best_dist < 0.05:
                return None

        del self.anchor_profiles[best_name]
        if best_name in self.expected_attendees:
            self.expected_attendees.remove(best_name)
        return best_name

    def _get_speaker_name(self, vector=None) -> str:
        """Priority: anchor match → expected attendee → Guest_NN."""
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

    def update_centroid(self, label: str, vector: np.ndarray, n_samples: int = 48000) -> None:
        """Weighted-average update; weight is capped so no single chunk dominates."""
        vec_flat = np.asarray(vector).flatten()
        if label not in self.centroids:
            return

        old = self.centroids[label]
        old_sample_count = self.centroid_sample_counts.get(label, 48000)
        new_total = old_sample_count + n_samples

        weight = n_samples / new_total
        weight = min(weight, 0.4)

        self.centroids[label] = (old * (1 - weight)) + (vec_flat * weight)
        self.centroid_sample_counts[label] = min(new_total, 240000)  # cap at 15s
        self.centroid_counts[label] = self.centroid_counts.get(label, 0) + 1

    def score_against_known(self, vector: np.ndarray, target_label: str) -> tuple[str, float]:
        """Score against a specific speaker; used for continuity checks on short segments."""
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

    def process_segment(
        self,
        vector: np.ndarray,
        allow_new_speaker: bool = True,
        dry_run: bool = False,
        prefer_label: str | None = None,
        n_samples: int = 48000,
    ) -> tuple[str, float]:
        """
        Main matching entry point. Returns (label, confidence).

        prefer_label wins ties within 2× margin (continuity enforcement).
        allow_new_speaker=False forces assignment to nearest known speaker.
        dry_run skips centroid updates.
        """
        vec_flat = np.asarray(vector).flatten()

        if not self.centroids:
            label = self._get_speaker_name(vector=vec_flat)
            if not dry_run:
                self.centroids[label] = vec_flat
                self.centroid_counts[label] = 1
                self.centroid_sample_counts[label] = n_samples
            return label, 1.0

        distances = self._all_distances(vec_flat)
        best_label, best_dist = distances[0]

        if prefer_label and prefer_label in self.centroids:
            prefer_dist = cosine(vec_flat, self.centroids[prefer_label])
            if prefer_dist - best_dist < self.margin * 2:
                best_label = prefer_label
                best_dist = prefer_dist

        confidence = max(0.0, 1.0 - best_dist)

        if len(distances) > 1:
            second_dist = distances[1][1]
            if second_dist - best_dist < self.margin:
                confidence *= 0.7

        if best_dist < self.threshold:
            if not dry_run:
                self.update_centroid(best_label, vec_flat, n_samples=n_samples)
            return best_label, confidence

        # Acoustic variation — same speaker, wider centroid spread.
        if best_dist < self.new_speaker_threshold:
            if not dry_run:
                self.update_centroid(best_label, vec_flat, n_samples=n_samples)
            return best_label, confidence * 0.8

        if not allow_new_speaker:
            return best_label, confidence * 0.5

        label = self._get_speaker_name(vector=vec_flat)
        if not dry_run:
            self.centroids[label] = vec_flat
            self.centroid_counts[label] = 1
            self.centroid_sample_counts[label] = n_samples
        return label, 1.0

    def get_finalized_centroids(self) -> dict[str, np.ndarray]:
        """
        Return live centroids, retroactively renaming Guest_NN entries if an
        unclaimed anchor matches closely enough. Runs once at meeting end.
        """
        result = dict(self.centroids)

        if self.anchor_profiles:
            guests = [n for n in result if n.startswith("Guest_")]
            for guest in guests:
                guest_vec = result[guest]
                best_anchor = None
                best_dist = float("inf")
                for anchor_name, anchor_vec in self.anchor_profiles.items():
                    dist = cosine(guest_vec, anchor_vec)
                    if dist < best_dist:
                        best_dist = dist
                        best_anchor = anchor_name
                if best_anchor and best_dist < 0.85:
                    result[best_anchor] = result.pop(guest)
                    del self.anchor_profiles[best_anchor]

        return result
