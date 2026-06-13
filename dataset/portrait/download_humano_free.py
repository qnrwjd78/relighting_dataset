from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))

from utils.util_progress import progress_bar, progress_write


FREE_SAMPLE_PAGE = "https://humano3d.com/free-sample/"
SINGLE_MODELS_PAGE = "https://humano3d.com/product-category/single-models/"
CHECKOUT_URL = "https://humano3d.com/checkout/"
AJAX_CHECKOUT_URL = "https://humano3d.com/?wc-ajax=checkout"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "humano_free"
DEFAULT_PREVIEW_ROOT = REPO_ROOT / "outputs" / "previews" / "humano_free"
USER_AGENT = "Mozilla/5.0 HumanoFreeDownloader/1.0"
SUPPORTED_EXTS = {".blend", ".fbx", ".obj", ".glb", ".gltf", ".dae", ".ply", ".stl"}

FALLBACK_PRODUCT_URLS = [
    "https://humano3d.com/product/new-3d-people-posed-free-model/",
    "https://humano3d.com/product/posed-free-model-01-11/",
    "https://humano3d.com/product/posed-arab-new-free-model/",
    "https://humano3d.com/product/posed-new-free-models/",
    "https://humano3d.com/product/animated-free-model/",
    "https://humano3d.com/product/rigged-rigged-plus-free-model/",
    "https://humano3d.com/product/posed-plus-free-model/",
    "https://humano3d.com/product/arab-animated-free-model/",
]


@dataclass(frozen=True)
class Product:
    title: str
    slug: str
    url: str
    product_id: str
    variations: list["Variation"]


@dataclass(frozen=True)
class Variation:
    variation_id: str
    format_label: str


@dataclass(frozen=True)
class Selection:
    product: Product
    variation: Variation


class ProductLinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {key.lower(): value for key, value in attrs}
        href = attr_map.get("href")
        if href and "/product/" in href:
            self.links.append(urljoin(self.base_url, html.unescape(href)))


class DownloadLinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.in_anchor = False
        self.current_href: str | None = None
        self.current_text: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {key.lower(): value for key, value in attrs}
        href = attr_map.get("href")
        if href and ("download_file=" in href or ".zip" in href.lower()):
            self.in_anchor = True
            self.current_href = urljoin(self.base_url, html.unescape(href))
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.in_anchor:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self.in_anchor and self.current_href:
            label = " ".join(" ".join(self.current_text).split())
            self.links.append((self.current_href, label))
            self.in_anchor = False
            self.current_href = None
            self.current_text = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover Humano free products and optionally download them by completing the free WooCommerce checkout. "
            "Use --dry-run first."
        )
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help=f"Output root. Default: {DEFAULT_OUT_DIR}")
    parser.add_argument(
        "--format",
        default="Blender",
        help='Preferred Humano file format. Examples: "Blender", "other", "objfbx", "fbx", "3ds Max". Default: Blender.',
    )
    parser.add_argument("--product", action="append", help="Only select products whose title or slug contains this text.")
    parser.add_argument("--product-url", action="append", help="Explicit Humano product URL. Can be repeated.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="List selected product variations without placing an order.")
    parser.add_argument("--discover-from-site", action="store_true", help="Discover free product links from Humano pages before fallback.")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded zips.")
    parser.add_argument("--extract-only", action="store_true", help="Extract existing zips from out-dir/zips; skip checkout/download.")
    parser.add_argument(
        "--delete-zip-after-extract",
        "--remove-zip-after-extract",
        dest="delete_zip_after_extract",
        action="store_true",
        help="Delete each zip only after successful extraction.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    parser.add_argument("--checkout-email", default=None, help="Email used for the free Humano checkout.")
    parser.add_argument("--account-username", default=None, help="Humano account username. Defaults to the email prefix.")
    parser.add_argument("--account-password", default=None, help="Humano account password for checkout account creation.")
    parser.add_argument("--billing-first-name", default=None)
    parser.add_argument("--billing-last-name", default=None)
    parser.add_argument("--billing-company", default=None)
    parser.add_argument("--billing-country", default=None, help="Two-letter country code, e.g. KR or US.")
    parser.add_argument("--billing-address1", default=None)
    parser.add_argument("--billing-city", default=None)
    parser.add_argument("--billing-postcode", default=None)
    parser.add_argument("--billing-state", default="")
    parser.add_argument(
        "--accept-terms",
        action="store_true",
        help="Required for automated checkout. Confirms that you accept Humano's checkout terms.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_PREVIEW_ROOT / "humano_free_objects.txt"),
        help="Manifest path written after extraction.",
    )
    parser.add_argument(
        "--metadata-out",
        default=str(DEFAULT_PREVIEW_ROOT / "humano_free_download_meta.json"),
        help="Download metadata JSON path.",
    )
    args = parser.parse_args()
    if args.delete_zip_after_extract and not (args.extract or args.extract_only):
        parser.error("--delete-zip-after-extract requires --extract or --extract-only")
    return args


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def request_with_retries(session: requests.Session, method: str, url: str, retries: int, timeout: float, **kwargs) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"Failed to {method.upper()} {url}: {last_error}")


def strip_html(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text).split())


def page_title(text: str, fallback: str) -> str:
    match = re.search(r"<h1[^>]*>(.*?)</h1>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return fallback
    return strip_html(match.group(1)) or fallback


def slug_from_url(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    return parts[-1] if parts else "humano-product"


def discover_product_urls(session: requests.Session, timeout: float, retries: int, include_site: bool, explicit: list[str] | None) -> list[str]:
    urls = list(explicit or [])
    if include_site:
        for page_url in (FREE_SAMPLE_PAGE, SINGLE_MODELS_PAGE):
            try:
                response = request_with_retries(session, "GET", page_url, retries, timeout)
            except Exception as exc:
                progress_write(f"[Humano] Could not discover from {page_url}: {exc}")
                continue
            parser = ProductLinkParser(page_url)
            parser.feed(response.text)
            urls.extend(parser.links)
    if not urls:
        urls.extend(FALLBACK_PRODUCT_URLS)
    urls = [url for url in urls if "/product/" in url]
    return sorted(dict.fromkeys(urls), key=slug_from_url)


def parse_product(session: requests.Session, url: str, timeout: float, retries: int) -> Product | None:
    response = request_with_retries(session, "GET", url, retries, timeout)
    if "€0" not in response.text and "&euro;0" not in response.text and "display_price&quot;:0" not in response.text:
        return None
    match = re.search(
        r'<form class="variations_form cart"[^>]*data-product_id="(\d+)"[^>]*data-product_variations="([^"]+)"',
        response.text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    product_id = match.group(1)
    raw_variations = json.loads(html.unescape(match.group(2)))
    variations = []
    for item in raw_variations:
        variation_id = item.get("variation_id")
        attrs = item.get("attributes") or {}
        format_label = attrs.get("attribute_choose-file-format")
        if variation_id and format_label:
            variations.append(Variation(str(variation_id), str(format_label)))
    if not variations:
        return None
    title = page_title(response.text, slug_from_url(url).replace("-", " ").title())
    return Product(title=title, slug=slug_from_url(url), url=url, product_id=product_id, variations=variations)


def normalize_format(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def variation_score(variation: Variation, wanted: str) -> tuple[int, str]:
    label = normalize_format(variation.format_label)
    want = normalize_format(wanted)
    aliases = {
        "blend": ["blender"],
        "bld": ["blender"],
        "blender": ["blender"],
        "obj": ["objfbx", "obj", "othersoftwareobjfbx", "othersoftwareobj"],
        "fbx": ["fbx", "objfbx", "othersoftwarefbx", "othersoftwareobjfbx", "othersoftwarefbxdae"],
        "dae": ["dae", "othersoftwarefbxdae"],
        "objfbx": ["objfbx", "othersoftwareobjfbx", "othersoftwareobj"],
        "other": ["othersoftwareobjfbx", "othersoftwarefbx", "othersoftwarefbxdae", "othersoftwareobj"],
        "3dsmax": ["3dsmax"],
        "max": ["3dsmax"],
        "cinema4d": ["cinema4d"],
        "c4d": ["cinema4d"],
        "sketchup": ["sketchup"],
        "skp": ["sketchup"],
        "rhino": ["rhino"],
        "maya": ["maya"],
        "twinmotion": ["twinmotion"],
    }
    candidates = aliases.get(want, [want])
    for index, candidate in enumerate(candidates):
        if label == candidate:
            return (index, variation.format_label)
        if candidate in label:
            return (index + len(candidates), variation.format_label)
    return (1000, variation.format_label)


def select_variation(product: Product, wanted_format: str) -> Variation | None:
    scored = sorted(product.variations, key=lambda variation: variation_score(variation, wanted_format))
    return scored[0] if scored and variation_score(scored[0], wanted_format)[0] < 1000 else None


def selected_products(products: list[Product], args: argparse.Namespace) -> list[Selection]:
    selected = products
    if args.product:
        needles = [item.lower() for item in args.product]
        selected = [
            product
            for product in selected
            if any(needle in product.title.lower() or needle in product.slug.lower() for needle in needles)
        ]
    selections: list[Selection] = []
    for product in selected:
        variation = select_variation(product, args.format)
        if variation is None:
            progress_write(f"[Humano] skip: no format matching {args.format!r} for {product.title}")
            continue
        selections.append(Selection(product, variation))
    if args.limit is not None:
        selections = selections[: args.limit]
    return selections


def validate_checkout_args(args: argparse.Namespace) -> None:
    if args.extract_only:
        return
    required = [
        "checkout_email",
        "account_password",
        "billing_first_name",
        "billing_last_name",
        "billing_country",
        "billing_address1",
        "billing_city",
        "billing_postcode",
    ]
    missing = [name for name in required if not getattr(args, name)]
    if not args.accept_terms:
        missing.append("accept_terms")
    if missing:
        pretty = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise SystemExit(
            "Humano does not expose direct zip links; automated download must complete the free checkout. "
            f"Missing required checkout option(s): {pretty}. Run with --dry-run to list the selected free products first."
        )


def add_to_cart(session: requests.Session, selection: Selection, timeout: float, retries: int) -> None:
    product = selection.product
    variation = selection.variation
    request_with_retries(session, "GET", product.url, retries, timeout)
    data = {
        "attribute_choose-file-format": variation.format_label,
        "quantity": "1",
        "add-to-cart": product.product_id,
        "product_id": product.product_id,
        "variation_id": variation.variation_id,
    }
    response = request_with_retries(session, "POST", product.url, retries, timeout, data=data, allow_redirects=True)
    text = strip_html(response.text)
    if "added to your cart" not in text.lower() and "장바구니" not in text:
        progress_write(f"[Humano] add-to-cart response did not include a normal success message: {product.title}")


def parse_checkout_nonce(text: str) -> str:
    match = re.search(r'name="woocommerce-process-checkout-nonce"\s+value="([^"]+)"', text)
    if not match:
        match = re.search(r'value="([^"]+)"\s+name="woocommerce-process-checkout-nonce"', text)
    if not match:
        raise RuntimeError("Could not find WooCommerce checkout nonce.")
    return html.unescape(match.group(1))


def default_username(email: str) -> str:
    prefix = email.split("@", 1)[0]
    username = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._-")
    return username or "humano_user"


def complete_checkout(session: requests.Session, args: argparse.Namespace) -> str:
    response = request_with_retries(session, "GET", CHECKOUT_URL, args.retries, args.timeout)
    nonce = parse_checkout_nonce(response.text)
    data = {
        "billing_email": args.checkout_email,
        "billing_first_name": args.billing_first_name,
        "billing_last_name": args.billing_last_name,
        "billing_company": args.billing_company or "",
        "billing_country": args.billing_country,
        "billing_address_1": args.billing_address1,
        "billing_address_2": "",
        "billing_postcode": args.billing_postcode,
        "billing_city": args.billing_city,
        "billing_state": args.billing_state or "",
        "order_comments": "Free Humano sample model download.",
        "createaccount": "1",
        "account_username": args.account_username or default_username(args.checkout_email),
        "account_password": args.account_password,
        "terms": "on",
        "terms-field": "1",
        "woocommerce-process-checkout-nonce": nonce,
        "_wp_http_referer": "/checkout/",
    }
    headers = {"Referer": CHECKOUT_URL, "X-Requested-With": "XMLHttpRequest"}
    response = request_with_retries(session, "POST", AJAX_CHECKOUT_URL, args.retries, args.timeout, data=data, headers=headers)
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Checkout did not return JSON: {strip_html(response.text)[:500]}") from exc
    if payload.get("result") != "success" or not payload.get("redirect"):
        messages = strip_html(str(payload.get("messages", payload)))
        raise RuntimeError(f"Checkout failed: {messages}")
    redirect = str(payload["redirect"])
    progress_write(f"[Humano] checkout complete: {redact_url(redirect)}")
    return redirect


def redact_url(url: str) -> str:
    parsed = urlparse(url)
    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in {"email", "order", "key"}:
            value = "***"
        query.append((key, value))
    return urlunparse(parsed._replace(query=urlencode(query)))


def parse_download_links(session: requests.Session, order_url: str, timeout: float, retries: int) -> list[tuple[str, str]]:
    response = request_with_retries(session, "GET", order_url, retries, timeout)
    parser = DownloadLinkParser(order_url)
    parser.feed(response.text)
    links = []
    seen = set()
    for url, label in parser.links:
        if url in seen:
            continue
        seen.add(url)
        links.append((url, label))
    if not links:
        raise RuntimeError("Could not find Humano download links on the order page.")
    return links


def filename_from_response(response: requests.Response, fallback: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition, flags=re.IGNORECASE)
    if match:
        return Path(html.unescape(match.group(1))).name
    parsed_name = Path(urlparse(response.url).path).name
    if parsed_name and "." in parsed_name:
        return parsed_name
    return fallback


def safe_filename(value: str) -> str:
    value = value.strip() or "humano_download.zip"
    return "".join(ch if ch.isalnum() or ch in {"_", "-", ".", " "} else "_" for ch in value).strip()


def safe_stem(value: str) -> str:
    return Path(safe_filename(value)).stem


def download_link(
    session: requests.Session,
    url: str,
    label: str,
    zip_dir: Path,
    *,
    chunk_size: int,
    timeout: float,
    retries: int,
    overwrite: bool,
) -> Path:
    zip_dir.mkdir(parents=True, exist_ok=True)
    fallback = safe_filename((label or "humano_download") + ".zip")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, stream=True, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                body = response.text[:1000]
                raise RuntimeError(f"download returned HTML instead of a file: {strip_html(body)[:300]}")
            filename = safe_filename(filename_from_response(response, fallback))
            if not filename.lower().endswith(".zip"):
                filename += ".zip"
            target = zip_dir / filename
            if target.exists() and not overwrite:
                progress_write(f"[Humano] exists: {target}")
                return target
            part = target.with_suffix(target.suffix + ".part")
            total = int(response.headers.get("content-length") or 0) or None
            with part.open("wb") as handle:
                with progress_bar(
                    total=total,
                    desc=filename,
                    unit="B",
                    leave=False,
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
                            pbar.update(len(chunk))
            part.replace(target)
            return target
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"Failed to download Humano file {redact_url(url)}: {last_error}")


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()) and not overwrite:
        progress_write(f"[Humano] extracted exists: {extract_dir}")
        return
    if extract_dir.exists() and overwrite:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)


def candidate_score(path: Path) -> tuple[int, int, str]:
    preferred = [".blend", ".fbx", ".glb", ".gltf", ".obj", ".dae", ".ply", ".stl"]
    ext_rank = preferred.index(path.suffix.lower()) if path.suffix.lower() in preferred else len(preferred)
    return (ext_rank, len(path.parts), str(path))


def find_asset_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    grouped: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            grouped.setdefault(path.parent, []).append(path)
    return sorted((sorted(paths, key=candidate_score)[0] for paths in grouped.values()), key=str)


def write_outputs(metadata: list[dict], extract_root: Path, manifest: Path, metadata_out: Path) -> None:
    assets = find_asset_files(extract_root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(path) for path in assets) + ("\n" if assets else ""), encoding="utf-8")
    metadata_out.write_text(
        json.dumps({"packages": metadata, "assets": [str(path) for path in assets]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    progress_write(f"[Humano] wrote manifest: {manifest}")
    progress_write(f"[Humano] wrote metadata: {metadata_out}")
    progress_write(f"[Humano] found assets: {len(assets)}")


def print_selection(selections: list[Selection]) -> None:
    progress_write(f"[Humano] Selected: {len(selections)} product(s)")
    for selection in selections:
        product = selection.product
        variation = selection.variation
        formats = ", ".join(item.format_label for item in product.variations)
        progress_write(
            f"  - {product.title}  product_id={product.product_id}  "
            f"variation_id={variation.variation_id}  format={variation.format_label}  formats=[{formats}]"
        )


def existing_zip_paths(zip_dir: Path) -> list[Path]:
    return sorted(zip_dir.glob("*.zip"))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    zip_dir = out_dir / "zips"
    extract_root = out_dir / "extracted"
    manifest = Path(args.manifest).resolve()
    metadata_out = Path(args.metadata_out).resolve()
    chunk_size = args.chunk_size_mb * 1024 * 1024

    session = make_session()
    metadata: list[dict] = []

    if args.extract_only:
        zip_paths = existing_zip_paths(zip_dir)
        if not zip_paths:
            raise SystemExit(f"No zips found under {zip_dir}")
        with progress_bar(zip_paths, total=len(zip_paths), desc="Humano extract", unit="zip") as pbar:
            for zip_path in pbar:
                extract_dir = extract_root / safe_stem(zip_path.name)
                progress_write(f"[Humano] Extract: {zip_path.name} -> {extract_dir}")
                extract_zip(zip_path, extract_dir, args.overwrite)
                if args.delete_zip_after_extract:
                    zip_path.unlink(missing_ok=True)
                metadata.append({"zip_path": str(zip_path), "extract_dir": str(extract_dir)})
        write_outputs(metadata, extract_root, manifest, metadata_out)
        return 0

    progress_write("[Humano] Discover free products")
    urls = discover_product_urls(session, args.timeout, args.retries, args.discover_from_site, args.product_url)
    products: list[Product] = []
    for url in urls:
        try:
            product = parse_product(session, url, args.timeout, args.retries)
        except Exception as exc:
            progress_write(f"[Humano] skip product page {url}: {exc}")
            continue
        if product is not None and product.title.lower().find("free") >= 0:
            products.append(product)
    products = sorted(dict((product.slug, product) for product in products).values(), key=lambda product: product.title)
    selections = selected_products(products, args)
    if not selections:
        raise SystemExit("No Humano free product variations matched the selected filters.")
    print_selection(selections)
    progress_write(f"[Humano] Output: {out_dir}")
    if args.dry_run:
        progress_write("[Humano] Dry run only. No cart, checkout, or download was performed.")
        return 0

    validate_checkout_args(args)

    with progress_bar(selections, total=len(selections), desc="Humano add cart", unit="item") as pbar:
        for selection in pbar:
            pbar.set_postfix(product=selection.product.slug[:24])
            progress_write(f"[Humano] Add to cart: {selection.product.title} ({selection.variation.format_label})")
            add_to_cart(session, selection, args.timeout, args.retries)

    order_url = complete_checkout(session, args)
    links = parse_download_links(session, order_url, args.timeout, args.retries)
    progress_write(f"[Humano] Download links found: {len(links)}")

    with progress_bar(links, total=len(links), desc="Humano downloads", unit="file") as pbar:
        for index, (url, label) in enumerate(pbar):
            pbar.set_postfix(file=(label or str(index + 1))[:24])
            try:
                zip_path = download_link(
                    session,
                    url,
                    label,
                    zip_dir,
                    chunk_size=chunk_size,
                    timeout=args.timeout,
                    retries=args.retries,
                    overwrite=args.overwrite,
                )
                extract_dir = None
                if args.extract:
                    extract_dir = extract_root / safe_stem(zip_path.name)
                    progress_write(f"[Humano] Extract: {zip_path.name} -> {extract_dir}")
                    extract_zip(zip_path, extract_dir, args.overwrite)
                    if args.delete_zip_after_extract:
                        zip_path.unlink(missing_ok=True)
                        progress_write(f"[Humano] Deleted zip: {zip_path}")
                metadata.append(
                    {
                        "label": label,
                        "zip_path": str(zip_path),
                        "extract_dir": str(extract_dir) if extract_dir else None,
                    }
                )
            except Exception as exc:
                if not args.skip_failed:
                    raise
                progress_write(f"[Humano] Failed: {label or redact_url(url)}: {exc}")
                metadata.append({"label": label, "status": "failed", "error": str(exc)})

    if args.extract:
        write_outputs(metadata, extract_root, manifest, metadata_out)
    else:
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps({"packages": metadata}, indent=2, ensure_ascii=False), encoding="utf-8")
        progress_write(f"[Humano] wrote metadata: {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
