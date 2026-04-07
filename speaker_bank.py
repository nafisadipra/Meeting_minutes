from scipy.spatial.distance import cosine

class SmartSpeakerBank:
    def __init__(self, attendees=None, threshold=0.4, alpha=0.1):
        self.centroids = {}  
        self.threshold = threshold
        self.alpha = alpha   
        self.next_id = 1
        # Store the list of names provided by the user
        self.attendees = attendees if attendees else []

    def _get_next_name(self):
        # If we still have real names in the list, use them
        if self.attendees:
            return self.attendees.pop(0) 
        # If a 3rd person speaks but we only gave 2 names, call them Guest
        else:
            name = f"Guest_{self.next_id:02d}"
            self.next_id += 1
            return name

    def process_segment(self, vector):
        vec_flat = vector.flatten()

        if not self.centroids:
            label = self._get_next_name()
            self.centroids[label] = vec_flat
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
            label = self._get_next_name()
            self.centroids[label] = vec_flat
            return label