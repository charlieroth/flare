defmodule Flare.XORModel do
  def model do
    x1_input = Axon.input("x1", shape: {nil, 1})
    x2_input = Axon.input("x2", shape: {nil, 1})

    x1_input
    |> Axon.concatenate(x2_input)
    |> Axon.dense(8)
    |> Axon.tanh()
    |> Axon.dense(1)
    |> Axon.sigmoid()
  end

  def training_data(batch_size \\ 32) do
    Stream.repeatedly(fn ->
      key = Nx.Random.key(42)
      {x1, _} = Nx.Random.normal(key, 0, 2, shape: {batch_size, 1}, type: :f32)
      {x2, _} = Nx.Random.normal(key, 0, 2, shape: {batch_size, 1}, type: :f32)
      y = Nx.logical_xor(x1, x2)

      {%{"x1" => x1, "x2" => x2}, y}
    end)
  end

  def training_params(model, data, epochs \\ 10) do
    model
    |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
    |> Axon.Loop.run(data, %{}, epochs: epochs, iterations: 1000)
  end

  def predict(model, training_params, %{"x1" => _x1_input, "x2" => _x2_input} = inputs) do
    Axon.predict(model, training_params, inputs)
  end
end
