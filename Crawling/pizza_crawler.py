import argparse

import crawler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-only", action="store_true")
    parser.parse_args()
    count = crawler.export_data_js()
    crawler.log(f"[MARKET] compatibility market export complete; display_items={count}")


if __name__ == "__main__":
    main()
