import sys

from .fetch import fetch_all


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m download <output_dir> <count>')
        sys.exit(1)

    output_dir = sys.argv[1]

    if len(sys.argv) > 2:
        count = int(sys.argv[2])
    else:
        count = 1000

    fetch_all(output_dir, count)


if __name__ == '__main__':
    main()
