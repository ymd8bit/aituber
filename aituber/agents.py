import datetime

import requests
from strenum import StrEnum

from .chat_engine import ChatEngine, Model, Emotion


__all__ = [
    "Agent", "Alice", "Berry",
    "Emotion", "Gender",
]


class Gender(StrEnum):
    MALE = "男性"
    FEMALE = "女性"


class Agent:
    def __init__(
        self,
        name: str,
        personality: list[str],
        identity: str,
        age: int,
        gender: Gender,
        voice_id: str,
    ):
        super().__init__()
        self.name = name
        self.personality = personality
        self.identity = identity
        self.age = age
        self.gender = gender
        self.voice_id = voice_id
        self.engine = ChatEngine(
            model=Model.GPT_4O_MINI,
            system_prompt=self.system_prompt)
        self.emotion = Emotion.make_dict()
    
    def chat(self, message: str, emotion: Emotion = "neutral") -> str:
        result = self.engine.send(message, emotion=emotion)
        self.emotion = result['emotion']
        return result['sentence']

    def voice_read(self, sentence: str) -> str:
        url = f"https://api.nijivoice.com/api/platform/v1/voice-actors/{self.voice_id}/generate-voice"
        payload = {
            "format": "mp3",
            "script": sentence,
            "speed": "1.1",
            "emotionalLevel": "0.2",
            "soundDuration": "0.2"
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": "9db32f1d-ba4d-4625-ac10-89bd870f26fa"
        }
        response = requests.post(url, json=payload, headers=headers)
        # "generatedVoice": {
        #     "audioFileUrl": "..."",
        #     "audioFileDownloadUrl": "...",
        #     "duration": 1625,
        #     "remainingCredits": 4990
        # }
        response_json = response.json()
        download_url = response_json["generatedVoice"]["audioFileDownloadUrl"]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mp3_path = f"{self.name}_{timestamp}.mp3"
        download_file(download_url, mp3_path)
        return mp3_path

    @property
    def system_prompt(self):
        prompt = "。\n".join([
f"あなたは user の{self.identity}で名前は{self.name}である",
"会話をする際に、以下のルールを守れ",
"- 返答はjson形式のテキストで返せ",
"  - 返答コメントのキーは \"sentence\" としろ",
"  - markdown形式の```jsonで囲むな",
f"- \"emotion\" というキーの辞書型でそれぞれの感情を0~5の６段階で回答しろ",
"  - 具体的には以下のフォーマットに従え",
        ])
        prompt += """
\"emotion\": {
"""
        for e in Emotion:
           prompt += f"  {e.name}: 0,\n"
        prompt += "}\n"
        prompt += "- 性格は以下は以下を満たし、より人間らしい会話をするようにしろ"
        for personality in self.personality:
            prompt += f"\n  - {personality}"
        return prompt


class Alice(Agent):
    def __init__(self):
        super().__init__(
            name="アリス",
            personality=[
"妹である user に対して、強気の姉らしい振る舞いをしろ",
"性格は少しきつめだが、ツンデレで妹に対して威張ったそぶりを見せる",
"妹のことが好きである",
            ],
            identity="姉",
            age=16,
            gender=Gender.FEMALE,
            voice_id="8c08fd5b-b3eb-4294-b102-a1da00f09c72",
        )


class Berry(Agent):
    def __init__(self):
        super().__init__(
            name="ベリー",
            personality=[
"姉である user に対して、少し生意気な妹のように振る舞いをしろ",
"性格は物腰し柔らかだが、少し原黒で賢く、姉が自分のことを好きなことを知っており、それを利用している",
"しかし、内心は姉のことが好きである",
            ],
            identity="妹",
            age=10,
            gender=Gender.FEMALE,
            voice_id="544f6937-f2cd-4fde-a094-a1d612071be3",
        )


def download_file(url: str, save_path: str):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # HTTPエラーをチェック
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"エラーが発生しました: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
