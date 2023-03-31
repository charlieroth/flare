import Config

config :nx, :default_backend, {Torchx.Backend, device: :mps}
