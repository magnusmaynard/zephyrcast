import keyring

def fetch_all_data():
    api_key = keyring.get_password("zephyr_api_key", "zephyrcast")
    print("run")