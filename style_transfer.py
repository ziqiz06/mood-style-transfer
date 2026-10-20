"""
Mood-based photo style transfer.

Pushes a content photo toward the averaged color/texture "mood" of a set of
reference images, using a frozen pretrained VGG16 as a feature extractor,
Gram matrices for style representation, and a weighted content/style loss
(Gatys et al., 2015).

Usage:
    python style_transfer.py --content examples/input.png --style-dir moods/dusk --output examples/output.png
    python style_transfer.py --content photo.jpg --style-dir moods/dusk --steps 1500 --show
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
from PIL import Image

# VGG16 layers used for style (multiple depths = texture/color at multiple
# scales) and for content (one mid-deep layer = structure, not exact pixels).
STYLE_LAYERS = {
    '0':  'conv1_1',
    '5':  'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '28': 'conv5_1',
}
CONTENT_LAYER = 'conv4_2'
LAYERS = {**STYLE_LAYERS, '21': CONTENT_LAYER}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_image(path, device, max_size=400):
    image = Image.open(path).convert('RGB')
    # Cap the longest side at max_size (transforms.Resize(int) would set the
    # shortest side instead, making portrait/landscape images larger than asked).
    scale = max_size / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0).to(device)


def tensor_to_image(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)


def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (c * h * w)  # normalize so scale doesn't depend on image size


def lowpass(tensor, sigma):
    kernel = int(2 * round(3 * sigma) + 1)  # cover +/- 3 sigma
    return TF.gaussian_blur(tensor, [kernel, kernel], [sigma, sigma])


def frequency_blend(stylized, content, sigma):
    """Low frequencies from the stylized image + high frequencies from the content.

    The mood (palette, tone, soft color shifts) is low frequency; the
    reference's textures (e.g. grass strands) are high frequency. Swapping in
    the content's own high frequencies keeps the mood while removing the
    copied texture and restoring the photo's original detail.
    """
    return lowpass(stylized, sigma) + (content - lowpass(content, sigma))


def collect_style_images(style_dir):
    style_dir = Path(style_dir)
    if not style_dir.is_dir():
        raise FileNotFoundError(f"Style directory not found: {style_dir}")
    paths = sorted(p for p in style_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No images found in style directory: {style_dir}")
    return paths


def compute_mood_style_grams(style_paths, model, device, max_size):
    """Average Gram matrices across every reference image in the set.

    This is the actual "learns the mood from a reference set" step: each
    image votes on the target's color/texture statistics at each layer, and
    the target is optimized toward the average, not any single image.
    """
    accumulated = {}
    for path in style_paths:
        style_image = load_image(path, device, max_size)
        style_features = get_features(style_image, model, STYLE_LAYERS)
        for layer, feat in style_features.items():
            accumulated.setdefault(layer, []).append(gram_matrix(feat))
    return {layer: torch.stack(grams).mean(dim=0) for layer, grams in accumulated.items()}


def build_vgg(device):
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(device)
    for param in vgg.parameters():
        param.requires_grad = False  # we only optimize the target image, not the network
    vgg.eval()
    return vgg


def run_style_transfer(content_path, style_dir, output_path, *, steps=2000,
                        style_weight=1e6, content_weight=1.0, max_size=400,
                        detail_sigma=8.0, device=None, show=False, log_every=200):
    device = device or get_device()
    print(f"Using device: {device}")

    vgg = build_vgg(device)

    content_image = load_image(content_path, device, max_size)
    content_features = get_features(content_image, vgg, LAYERS)

    style_paths = collect_style_images(style_dir)
    print(f"Building mood signature from {len(style_paths)} reference image(s) in '{style_dir}':")
    for p in style_paths:
        print(f"  - {p.name}")
    style_grams = compute_mood_style_grams(style_paths, vgg, device, max_size)

    target = content_image.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([target], lr=0.003)

    for step in range(steps):
        target_features = get_features(target, vgg, LAYERS)

        content_loss = torch.mean((target_features[CONTENT_LAYER] - content_features[CONTENT_LAYER]) ** 2)

        style_loss = 0.0
        for layer, style_gram in style_grams.items():
            target_gram = gram_matrix(target_features[layer])
            style_loss = style_loss + torch.mean((target_gram - style_gram) ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:4d}/{steps}  loss={total_loss.item():10.2f}  "
                  f"content={content_loss.item():.4f}  style={style_loss.item():.6f}")

    result = target.detach()
    if detail_sigma > 0:
        result = frequency_blend(result, content_image, detail_sigma)
    output_image = tensor_to_image(result)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)
    print(f"Saved result to {output_path}")

    if show:
        import matplotlib.pyplot as plt
        plt.imshow(output_image)
        plt.title("Output")
        plt.axis("off")
        plt.show()

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content", required=True, help="Path to the content photo")
    parser.add_argument("--style-dir", required=True,
                         help="Folder of reference images defining the target mood")
    parser.add_argument("--output", default="output.png", help="Where to save the result")
    parser.add_argument("--steps", type=int, default=2000, help="Optimization steps")
    parser.add_argument("--style-weight", type=float, default=1e6)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--max-size", type=int, default=400, help="Longest image side, in pixels")
    parser.add_argument("--detail-sigma", type=float, default=8.0,
                         help="Gaussian sigma for keeping only the mood's low frequencies and the "
                              "photo's own detail (0 = off, raw style transfer output)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"],
                         help="Force a device (default: auto-detect best available)")
    parser.add_argument("--show", action="store_true", help="Display the result when done")
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    run_style_transfer(
        args.content, args.style_dir, args.output,
        steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        max_size=args.max_size,
        detail_sigma=args.detail_sigma,
        device=get_device(args.device),
        show=args.show,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
