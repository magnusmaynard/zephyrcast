import keyring
import requests
import os
from tqdm import tqdm
import asyncio
import aiohttp
from zephyrcast import project_config


def _get_download_urls(url: str, key: str) -> list[str]:
    response = requests.get(
        url,
        params={"key": key},
    )

    response.raise_for_status()

    return [item["url"] for item in response.json()]


async def _download_file(
    session: aiohttp.ClientSession, url: str, dst: str, bar: tqdm
) -> bool:
    filename = os.path.basename(url)
    filepath = os.path.join(dst, filename)

    try:
        async with session.get(url) as response:
            file_data = await response.read()

            with open(filepath, "wb") as f:
                f.write(file_data)
            bar.update(1)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        bar.update(1)
        return False


async def _fetch_all_data():
    api_key = keyring.get_password("zephyr_api_key", "zephyrcast")
    api_url = "https://api.zephyrapp.nz/v1/json-output"
    download_dir = project_config["data_dir"]

    print("Preparing download...")
    urls = _get_download_urls(url=api_url, key=api_key)

    os.makedirs(download_dir, exist_ok=True)

    existing_files = set(os.listdir(download_dir))

    download_files = []
    for url in urls:
        filename = os.path.basename(url)
        if filename not in existing_files:
            download_files.append(url)

    print(f"{len(urls)} total files found")
    print(f"{len(existing_files)} downloaded")
    print(f"{len(download_files)} to download")

    with tqdm(total=len(download_files), desc="Downloading", unit="file") as bar:
        async with aiohttp.ClientSession() as session:
            tasks = [
                _download_file(session=session, url=url, dst=download_dir, bar=bar)
                for download_url in download_files
            ]

            result = await asyncio.gather(*tasks)


def fetch():
    asyncio.run(_fetch_all_data())
    print("Download complete")
