import pathlib
import json
from knockknock import telegram_sender


def notify(key_path=None):

    if key_path is None:
        key_path = pathlib.Path(__file__).parent / 'telegram.json'
    key_path = pathlib.Path(key_path)

    if key_path.exists():
        with open(key_path, mode='r') as infile:
            json_file = json.load(infile)

        chat_id = json_file['chat-id']
        token = json_file['token']

        return telegram_sender(token, chat_id)
    else:
        def bypass(func):
            return func
        return bypass
