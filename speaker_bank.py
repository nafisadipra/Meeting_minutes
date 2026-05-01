from scipy.spatial.distance import cosine

class SmartSpeakerBank:
    # anchor_profiles is a dictionary like: {"Nafis": [0.12, 0.45, ...], "Muyeed": [0.88, 0.21, ...]}
    def __init__(self, anchor_profiles=None, threshold=0.75, alpha=0.25):
        # Start the bank with the pre-enrolled voice profiles!
        self.centroids = anchor_profiles if anchor_profiles else {}
        self.threshold = threshold
        self.alpha = alpha   
        self.next_guest_id = 1

    def _get_guest_name(self):
        name = f"Guest_{self.next_guest_id:02d}"
        self.next_guest_id += 1
        return name

    def process_segment(self, vector):
        vec_flat = vector.flatten()

        # If we have no profiles at all, the first person speaking is Guest_01
        if not self.centroids:
            label = self._get_guest_name()
            self.centroids[label] = vec_flat
            return label

        best_label = None
        min_dist = float('inf')

        # Compare the incoming voice to all our enrolled profiles (and previous guests)
        for label, centroid in self.centroids.items():
            dist = cosine(vec_flat, centroid)
            if dist < min_dist:
                min_dist = dist
                best_label = label

        # If it's a match, update the profile slightly to adapt to their microphone today
        if min_dist < self.threshold:
            old_centroid = self.centroids[best_label]
            new_centroid = (old_centroid * (1 - self.alpha)) + (vec_flat * self.alpha)
            self.centroids[best_label] = new_centroid
            return best_label
        else:
            # If the voice doesn't match Nafis or Muyeed, it's a new guest!
            label = self._get_guest_name()
            self.centroids[label] = vec_flat
            return label