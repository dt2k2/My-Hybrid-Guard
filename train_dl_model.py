import sys

from train_url_model import main


if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "dl"])
    main()
