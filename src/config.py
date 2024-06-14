from dataclasses import dataclass


@dataclass
class Config:
    ensemble_size: int = 60  # デフォルト値を設定
    number_of_prior: int = 1
    number_of_obs: int = 200
    noise_flag: bool = False


# インスタンス化して設定を作成
config = Config()
