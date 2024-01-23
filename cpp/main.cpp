#include <generators.h>
#include <iostream>

int main() {

  std::cout << "Initializing OnnxRuntime...";
  std::cout.flush();
  Ort::InitApi();
  std::unique_ptr<OrtEnv> g_ort_env = OrtEnv::Create();
  std::cout << "done" << std::endl;

  std::cout << "Running phi-2\r\n";

  auto prompt = R"(
def print_prime(n):
'''
Print all primes between 1 and n
'''
)";

  std::cout << "With prompt:" << prompt << "\r\n";

  auto provider_options = Generators::GetDefaultProviderOptions(Generators::DeviceType::CUDA);
  auto model = Generators::CreateModel(*g_ort_env, "../models/microsoft/phi-2/int4", &provider_options);
  auto tokenizer = model->CreateTokenizer();
  auto tokens = tokenizer->Encode(prompt);

  Generators::SearchParams params{*model};
  params.batch_size = 1;
  params.sequence_length = static_cast<int>(tokens.size());
  params.input_ids = tokens;
  params.max_length = 128;

  auto search = params.CreateSearch();
  auto result=model->Generate(params);

  std::cout << tokenizer->Decode(result) << "\r\n";
  std::cout << "Test complete\r\n";
}