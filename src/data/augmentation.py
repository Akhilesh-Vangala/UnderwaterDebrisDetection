import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Tuple, List, Optional
import random
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UnderwaterHazeSimulation:
    def __init__(self, beta_range: Tuple[float, float] = (0.5, 1.5), 
                 atmospheric_light_range: Tuple[float, float] = (0.6, 0.9)):
        self.beta_range = beta_range
        self.atmospheric_light_range = atmospheric_light_range
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        beta = random.uniform(*self.beta_range)
        h, w = image.shape[:2]
        
        A = np.array([
            random.uniform(*self.atmospheric_light_range),
            random.uniform(*self.atmospheric_light_range),
            random.uniform(*self.atmospheric_light_range)
        ])
        
        depth_map = self._generate_depth_map(h, w)
        t = np.exp(-beta * depth_map)
        t = np.expand_dims(t, axis=2)
        
        hazy = image.astype(np.float32) / 255.0 * t + A * (1 - t)
        hazy = np.clip(hazy * 255, 0, 255).astype(np.uint8)
        return hazy
    
    def _generate_depth_map(self, h: int, w: int) -> np.ndarray:
        depth = np.random.rand(h, w)
        depth = gaussian_filter(depth, sigma=random.uniform(5, 15))
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth


class UnderwaterColorJitter:
    def __init__(self, brightness: float = 0.3, contrast: float = 0.3, 
                 saturation: float = 0.3, hue: float = 0.1, 
                 channel_shift_range: float = 20.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.channel_shift_range = channel_shift_range
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < 0.6:
            alpha = 1 + random.uniform(-self.brightness, self.brightness)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        
        if random.random() < 0.6:
            alpha = 1 + random.uniform(-self.contrast, self.contrast)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        
        if random.random() < 0.6:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= (1 + random.uniform(-self.saturation, self.saturation))
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-self.hue * 180, self.hue * 180)) % 180
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        if random.random() < 0.4:
            for c in range(3):
                shift = random.uniform(-self.channel_shift_range, self.channel_shift_range)
                image[:, :, c] = np.clip(image[:, :, c].astype(np.float32) + shift, 0, 255).astype(np.uint8)
        
        return image


class UnderwaterContrastAdjustment:
    def __init__(self, alpha_range: Tuple[float, float] = (0.5, 1.5),
                 beta_range: Tuple[int, int] = (-30, 30),
                 gamma_range: Tuple[float, float] = (0.7, 1.3)):
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < 0.7:
            alpha = random.uniform(*self.alpha_range)
            beta = random.randint(*self.beta_range)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        if random.random() < 0.5:
            gamma = random.uniform(*self.gamma_range)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            image = cv2.LUT(image, table)
        
        return image


class UnderwaterCLAHE:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8),
                 adaptive: bool = True):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.adaptive = adaptive
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < 0.6:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            if self.adaptive:
                clip_limit = random.uniform(self.clip_limit * 0.5, self.clip_limit * 1.5)
            else:
                clip_limit = self.clip_limit
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=self.tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return image


class UnderwaterNoiseInjection:
    def __init__(self, noise_type: str = 'gaussian', gaussian_std: float = 10.0,
                 salt_pepper_prob: float = 0.01):
        self.noise_type = noise_type
        self.gaussian_std = gaussian_std
        self.salt_pepper_prob = salt_pepper_prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            if self.noise_type == 'gaussian':
                noise = np.random.normal(0, self.gaussian_std, image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            elif self.noise_type == 'salt_pepper':
                h, w, c = image.shape
                num_salt = np.ceil(self.salt_pepper_prob * h * w).astype(int)
                coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
                image[coords[0], coords[1], :] = 255
                num_pepper = np.ceil(self.salt_pepper_prob * h * w).astype(int)
                coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
                image[coords[0], coords[1], :] = 0
        return image


class UnderwaterBlur:
    def __init__(self, blur_types: List[str] = ['gaussian', 'motion', 'defocus'],
                 kernel_size_range: Tuple[int, int] = (3, 7)):
        self.blur_types = blur_types
        self.kernel_size_range = kernel_size_range
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < 0.4:
            blur_type = random.choice(self.blur_types)
            ksize = random.choice(range(self.kernel_size_range[0], self.kernel_size_range[1] + 1, 2))
            
            if blur_type == 'gaussian':
                image = cv2.GaussianBlur(image, (ksize, ksize), 0)
            elif blur_type == 'motion':
                kernel = np.zeros((ksize, ksize))
                kernel[int((ksize-1)/2), :] = np.ones(ksize)
                kernel = kernel / ksize
                image = cv2.filter2D(image, -1, kernel)
            elif blur_type == 'defocus':
                image = cv2.GaussianBlur(image, (ksize, ksize), random.uniform(0.5, 2.0))
        return image


class UnderwaterGeometricTransform:
    def __init__(self, rotation_range: Tuple[int, int] = (-15, 15),
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 shear_range: Tuple[float, float] = (-5, 5)):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            angle = random.uniform(*self.rotation_range)
            scale = random.uniform(*self.scale_range)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        return image


class UnderwaterCutMix:
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, image1: np.ndarray, image2: np.ndarray, 
                 label1: int, label2: int) -> Tuple[np.ndarray, int, float]:
        if random.random() > self.prob:
            return image1, label1, 1.0
        
        lam = np.random.beta(self.alpha, self.alpha)
        h, w = image1.shape[:2]
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)
        
        cy = np.random.randint(h)
        cx = np.random.randint(w)
        
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        
        image1[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
        
        return image1, label1, lam


class UnderwaterMixup:
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, image1: np.ndarray, image2: np.ndarray,
                 label1: int, label2: int) -> Tuple[np.ndarray, int, float]:
        if random.random() > self.prob:
            return image1, label1, 1.0
        
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_image = (lam * image1.astype(np.float32) + 
                      (1 - lam) * image2.astype(np.float32)).astype(np.uint8)
        return mixed_image, label1, lam


class UnderwaterAugmentation:
    def __init__(self, img_size: int = 224, use_haze: bool = True,
                 use_color_jitter: bool = True, use_contrast: bool = True,
                 use_clahe: bool = True, use_noise: bool = True,
                 use_blur: bool = True, use_geometric: bool = True,
                 use_cutmix: bool = False, use_mixup: bool = False):
        self.img_size = img_size
        self.use_cutmix = use_cutmix
        self.use_mixup = use_mixup
        
        self.transforms_list = []
        
        if use_haze:
            self.transforms_list.append(UnderwaterHazeSimulation())
        if use_color_jitter:
            self.transforms_list.append(UnderwaterColorJitter())
        if use_contrast:
            self.transforms_list.append(UnderwaterContrastAdjustment())
        if use_clahe:
            self.transforms_list.append(UnderwaterCLAHE())
        if use_noise:
            self.transforms_list.append(UnderwaterNoiseInjection())
        if use_blur:
            self.transforms_list.append(UnderwaterBlur())
        if use_geometric:
            self.transforms_list.append(UnderwaterGeometricTransform())
        
        self.geometric_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if use_cutmix:
            self.cutmix = UnderwaterCutMix()
        if use_mixup:
            self.mixup = UnderwaterMixup()
    
    def __call__(self, image: np.ndarray, label: Optional[int] = None,
                 second_image: Optional[np.ndarray] = None,
                 second_label: Optional[int] = None) -> torch.Tensor:
        if len(self.transforms_list) > 0 and random.random() < 0.8:
            num_transforms = random.randint(1, min(3, len(self.transforms_list)))
            selected_transforms = random.sample(self.transforms_list, num_transforms)
            for transform in selected_transforms:
                image = transform(image)
        
        if self.use_cutmix and second_image is not None and second_label is not None:
            image, label, lam = self.cutmix(image, second_image, label, second_label)
        elif self.use_mixup and second_image is not None and second_label is not None:
            image, label, lam = self.mixup(image, second_image, label, second_label)
        
        image = self.geometric_transforms(image)
        return image


class UnderwaterValidationTransform:
    def __init__(self, img_size: int = 224):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return self.transform(image)


class TestTimeAugmentation:
    def __init__(self, img_size: int = 224, num_augmentations: int = 5):
        self.img_size = img_size
        self.num_augmentations = num_augmentations
        
        self.augmentations = [
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.VerticalFlip(p=1.0),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Rotate(limit=15, p=1.0),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        ]
    
    def __call__(self, image: np.ndarray) -> List[torch.Tensor]:
        augmented_images = []
        for aug in self.augmentations[:self.num_augmentations]:
            aug_img = aug(image=image)['image']
            augmented_images.append(aug_img)
        return augmented_images
