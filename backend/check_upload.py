"""Upload a JPEG to the running backend without PowerShell curl quoting issues."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser(description="Smoke test /upload_raw")
    p.add_argument("--file", required=True, type=Path)
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--page", default="P001")
    p.add_argument("--condition", default="C01_normal")
    args = p.parse_args()

    if not args.file.is_file():
        raise FileNotFoundError(args.file)
    query = urlencode({"page_id": args.page, "condition": args.condition})
    url = f"{args.server.rstrip('/')}/upload_raw?{query}"
    request = Request(url, data=args.file.read_bytes(), method="POST",
                      headers={"Content-Type": "image/jpeg"})
    try:
        with urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        print("HTTP error", e.code, e.read().decode("utf-8", errors="replace"))
        raise
    except URLError as e:
        print("Cannot reach backend:", e)
        raise
    print(json.dumps(body, ensure_ascii=False, indent=2))
    print(f"View image in browser: {args.server.rstrip('/')}{body['view_url']}")


if __name__ == "__main__":
    main()
