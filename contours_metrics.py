class ContourMetrics:
    """
    Class to evaluate contour detection performance
    """
    
    @staticmethod
    def calculate_metrics(contours, image, expected_count=None, ground_truth=None):
        """
        Calculate multiple metrics for contour detection performance
        
        Parameters:
        contours (list): List of detected contours
        image (numpy.ndarray): Original input image
        expected_count (int, optional): Expected number of puzzle pieces
        ground_truth (list, optional): Ground truth contours if available
        
        Returns:
        dict: Dictionary of metrics
        """
        import cv2
        import numpy as np
        
        metrics = {}
        
        # 1. Count-based metrics
        metrics['detected_count'] = len(contours)
        metrics['expected_count'] = expected_count
        if expected_count:
            metrics['detection_rate'] = len(contours) / expected_count
        
        # 2. Area-based metrics
        total_area = image.shape[0] * image.shape[1]
        contour_areas = [cv2.contourArea(cnt) for cnt in contours]
        metrics['mean_area'] = np.mean(contour_areas) if contours else 0
        metrics['std_area'] = np.std(contour_areas) if contours else 0
        metrics['min_area'] = np.min(contour_areas) if contours else 0
        metrics['max_area'] = np.max(contour_areas) if contours else 0
        metrics['total_piece_area'] = sum(contour_areas)
        metrics['area_coverage'] = sum(contour_areas) / total_area
        
        # 3. Shape metrics
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
        compactness = [4 * np.pi * area / (perim**2) if perim > 0 else 0 
                      for area, perim in zip(contour_areas, perimeters)]
        metrics['mean_compactness'] = np.mean(compactness) if compactness else 0
        metrics['std_compactness'] = np.std(compactness) if compactness else 0
        
        # 4. Convexity metrics
        convexity = []
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            cnt_area = cv2.contourArea(cnt)
            convexity.append(cnt_area / hull_area if hull_area > 0 else 0)
        
        metrics['mean_convexity'] = np.mean(convexity) if convexity else 0
        metrics['std_convexity'] = np.std(convexity) if convexity else 0
        
        # 5. Edge alignment metrics (how well contours align with actual edges)
        edge_map = cv2.Canny(image, 50, 150)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, 2)
        
        # Calculate overlap between detected contours and edge map
        overlap = cv2.bitwise_and(edge_map, mask)
        metrics['edge_alignment'] = np.sum(overlap > 0) / (np.sum(mask > 0) + 1e-6)
        
        # 6. Ground truth metrics (if available)
        if ground_truth and len(ground_truth) > 0:
            # IoU-based metrics
            gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(gt_mask, ground_truth, -1, 255, -1)
            
            pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(pred_mask, contours, -1, 255, -1)
            
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            metrics['iou'] = intersection / union if union > 0 else 0
            
            # F1 score components
            tp = intersection
            fp = np.sum(pred_mask) - intersection
            fn = np.sum(gt_mask) - intersection
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return metrics

    @staticmethod
    def visualize_metrics(image, contours, metrics, output_path="debug/metrics_visualization.jpg"):
        """
        Create a visualization of the metrics and detected contours
        
        Parameters:
        image (numpy.ndarray): Original input image
        contours (list): List of detected contours
        metrics (dict): Dictionary of metrics from calculate_metrics
        output_path (str): Path to save visualization
        
        Returns:
        numpy.ndarray: Visualization image
        """
        import cv2
        import numpy as np
        
        # Create visualization image
        vis_img = image.copy()
        
        # Draw contours with color based on area (normalized to a color scale)
        areas = [cv2.contourArea(cnt) for cnt in contours]
        if areas:
            min_area, max_area = min(areas), max(areas)
            area_range = max_area - min_area
            
            for i, cnt in enumerate(contours):
                if area_range > 0:
                    # Normalize area to 0-1 range
                    norm_area = (areas[i] - min_area) / area_range
                    # Create color: smaller pieces are more red, larger are more green
                    color = (int(255 * (1 - norm_area)), int(255 * norm_area), 0)
                else:
                    color = (0, 255, 0)
                
                cv2.drawContours(vis_img, [cnt], 0, color, 2)
                
                # Add contour number
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(vis_img, f"{i}", (cx, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Create metrics panel
        metrics_panel = np.ones((300, image.shape[1], 3), dtype=np.uint8) * 255
        
        # Add metrics text
        metrics_text = [
            f"Detected Pieces: {metrics.get('detected_count', 'N/A')}",
            f"Expected Pieces: {metrics.get('expected_count', 'N/A')}",
            f"Detection Rate: {metrics.get('detection_rate', 'N/A'):.2f}" if 'detection_rate' in metrics else "",
            f"Mean Area: {metrics.get('mean_area', 'N/A'):.1f}",
            f"Area Std Dev: {metrics.get('std_area', 'N/A'):.1f}",
            f"Area Coverage: {metrics.get('area_coverage', 'N/A'):.2f}",
            f"Edge Alignment: {metrics.get('edge_alignment', 'N/A'):.2f}",
            f"Mean Compactness: {metrics.get('mean_compactness', 'N/A'):.2f}",
            f"Mean Convexity: {metrics.get('mean_convexity', 'N/A'):.2f}",
        ]
        
        if 'iou' in metrics:
            metrics_text.extend([
                f"IoU: {metrics['iou']:.4f}",
                f"Precision: {metrics['precision']:.4f}",
                f"Recall: {metrics['recall']:.4f}",
                f"F1 Score: {metrics['f1_score']:.4f}"
            ])
        
        y = 30
        for text in metrics_text:
            if text:  # Only add non-empty text
                cv2.putText(metrics_panel, text, (20, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y += 25
        
        # Combine visualization and metrics panel
        result = np.vstack((vis_img, metrics_panel))
        
        # Save result
        cv2.imwrite(output_path, result)
        
        return result
    
    @staticmethod
    def generate_report(metrics, output_path="debug/metrics_report.txt"):
        """
        Generate a text report of the metrics
        
        Parameters:
        metrics (dict): Dictionary of metrics from calculate_metrics
        output_path (str): Path to save report
        """
        import json
        
        # Format metrics for better readability
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if isinstance(value, int):
                    formatted_metrics[key] = value
                else:
                    formatted_metrics[key] = round(value, 4)
            else:
                formatted_metrics[key] = value
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(formatted_metrics, f, indent=4)
            
        return formatted_metrics