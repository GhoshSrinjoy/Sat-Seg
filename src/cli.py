"""Installed command-line entry point for the local application and models."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="sat-clas")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the local map interface")
    serve.add_argument("--port", type=int, default=8765)
    search = commands.add_parser("search", help="Look up address/place candidates")
    search.add_argument("query")
    features = commands.add_parser("features", help="Query mapped features in a bounding box")
    features.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    features.add_argument("--category", default="buildings")
    features.add_argument("--tag")
    segment = commands.add_parser("segment", help="Segment an image using local SAM 3")
    segment.add_argument("image", type=Path)
    segment.add_argument("--prompt")
    segment.add_argument("--threshold", type=float)
    segment.add_argument("--bands", nargs=3, type=int)
    segment.add_argument("--bounds", nargs=4, type=float)
    segment.add_argument("--window-origin", nargs=2, type=int)
    segment.add_argument("--enrich", action="store_true")
    compare = commands.add_parser("compare", help="Compare before/after PNG, JPEG or georeferenced images")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--prompt")
    compare.add_argument("--bands", nargs=3, type=int)
    compare.add_argument("--alignment", choices=["auto", "same_grid"])
    compare.add_argument("--bounds", nargs=4, type=float, help="Full north-up After image extent")
    train = commands.add_parser("train", help="Train an RGB tile classifier from labeled data")
    train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=.001)
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--device", default="cuda")
    classify = commands.add_parser("classify", help="Predict an image class using a trained checkpoint")
    classify.add_argument("image", type=Path)
    classify.add_argument("--checkpoint", required=True, type=Path)
    classify.add_argument("--device", default="cuda")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "serve":
        import uvicorn
        uvicorn.run("src.interface.app:app", host="127.0.0.1", port=args["port"])
        return
    if command == "search":
        from src.geo.providers import search_address
        result = search_address(args["query"])
    elif command == "features":
        from src.geo.providers import query_features
        result = query_features(args["bbox"], args["category"], args["tag"])
    elif command == "segment":
        from src.inference.segmentation import segment_image
        result = segment_image(args.pop("image"), **args)
    elif command == "compare":
        from src.inference.change_detection import compare_images
        result = compare_images(args.pop("before"), args.pop("after"), **args)
    elif command == "train":
        from src.training.train_classifier import train_classifier
        result = train_classifier(**args)
    else:
        from src.training.train_classifier import classify_image
        result = classify_image(args["checkpoint"], args["image"], args["device"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
