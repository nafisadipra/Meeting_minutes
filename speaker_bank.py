from scipy.spatial.distance import cosine

class SmartSpeakerBank:
    def __init__(self, threshold=0.4, alpha=0.1):
        self.centroids = {}  
        self.threshold = threshold
        self.alpha = alpha   
        self.next_id = 1

    def process_segment(self, vector):
        vec_flat = vector.flatten()

        if not self.centroids:
            label = f"Speaker_{self.next_id:02d}"
            self.centroids[label] = vec_flat
            self.next_id += 1
            return label

        best_label = None
        min_dist = float('inf')

        for label, centroid in self.centroids.items():
            dist = cosine(vec_flat, centroid)
            if dist < min_dist:
                min_dist = dist
                best_label = label

        if min_dist < self.threshold:
            old_centroid = self.centroids[best_label]
            new_centroid = (old_centroid * (1 - self.alpha)) + (vec_flat * self.alpha)
            self.centroids[best_label] = new_centroid
            return best_label
        else:
            label = f"Speaker_{self.next_id:02d}"
            self.centroids[label] = vec_flat
            self.next_id += 1
            return label