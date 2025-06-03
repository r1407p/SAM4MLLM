import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import List, Tuple

class PointGenerator:
    def __init__(self):
        self.sam_ckpt = "./semantic_1/sam_vit_h_4b8939.pth"
        self.sam = sam_model_registry["default"](checkpoint=self.sam_ckpt).to(device='cuda')
        
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    @staticmethod
    def generate_point_grid(image_data, bbox: List[float]) -> List[List[float]]:
        """
        Generate a grid of points within the bounding box.
        :param image_data: bytes of the image
        :param bbox: bounding box in the format [x1, y1, x2, y2]
        :return: list of points in the format [[x1, y1], [x2, y2], ...]
        """
        points = [
            [93, 70], [51, 89], [91, 90], [32, 32], [88, 10],
            [12, 28], [29, 52], [49, 49], [28, 12], [59, 60],
            [9, 48], [52, 29], [31, 92], [68, 13], [73, 73]
        ]
        return points

    def generate_edge_points(self, image_data, bbox: list[int],
                        num_pairs: int = 10, offset: int = 15) -> list[list[int]]:
        """
        Args:
            image: np.ndarray (H, W, 3) full image
            bbox: [x1, y1, x2, y2] in 0~1000 relative coordinates
            num_pairs: number of edge point pairs (inside + outside)
            offset: number of pixels to move along normal for each side

        Returns:
            List of [x_pct, y_pct] points in 0~100 relative to full image size
        """
        sam_ckpt = "./semantic_1/sam_vit_h_4b8939.pth"  # download from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
        
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        H, W = image.shape[:2]
        x1, y1, x2, y2 = [int(coord / 1000 * size) for coord, size in zip(bbox, [W, H, W, H])]
        cropped = image[y1:y2, x1:x2]

        # Step 1: Segmentation
        masks = self.mask_generator.generate(cropped)
        if len(masks) == 0:
            return []

        # Step 2: Select largest mask
        masks.sort(key=lambda m: np.sum(m['segmentation']), reverse=True)
        seg_mask = masks[0]['segmentation'].astype(np.uint8) * 255

        # Step 3: Compute gradient
        blurred_mask = cv2.GaussianBlur(seg_mask, (5, 5), 0)
        sobelx = cv2.Sobel(blurred_mask, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(blurred_mask, cv2.CV_64F, 0, 1, ksize=5)
        edges = cv2.Canny(blurred_mask, 100, 200)
        ys, xs = np.where(edges > 0)
        if len(xs) == 0:
            return []

        # Step 4: Sample edge points and compute normal offsets
        indices = np.random.choice(len(xs), size=min(num_pairs, len(xs)), replace=False)
        inside_points, outside_points = [], []

        for i in indices:
            x_edge, y_edge = xs[i], ys[i]
            gx = sobelx[y_edge, x_edge]
            gy = sobely[y_edge, x_edge]
            norm = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
            nx, ny = gx / norm, gy / norm  # corrected normal direction

            x_in = int(np.clip(x_edge - nx * offset, 0, x2 - x1 - 1))
            y_in = int(np.clip(y_edge - ny * offset, 0, y2 - y1 - 1))
            x_out = int(np.clip(x_edge + nx * offset, 0, x2 - x1 - 1))
            y_out = int(np.clip(y_edge + ny * offset, 0, y2 - y1 - 1))

            inside_points.append([x_in + x1, y_in + y1])
            outside_points.append([x_out + x1, y_out + y1])

        all_points = inside_points + outside_points
        # 將點轉為 bbox 內的相對百分比 [0~100]
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        percent_pts = [
            [
                int((x - x1) / bbox_w * 100),
                int((y - y1) / bbox_h * 100)
            ]
            for x, y in all_points
        ]

        vis_save_path = 'ntu_final_project/output_edge_points_vis.png'
        if vis_save_path:
            vis_img = image.copy()

            mask_overlay = np.zeros_like(image)
            mask_color = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
            mask_overlay[y1:y2, x1:x2] = mask_color
            vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.5, 0)

            for x, y in inside_points:
                cv2.circle(vis_img, (x, y), 3, (0, 0, 255), -1)
            for x, y in outside_points:
                cv2.circle(vis_img, (x, y), 3, (255, 0, 0), -1)

            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(vis_save_path, vis_img)

        return percent_pts



            