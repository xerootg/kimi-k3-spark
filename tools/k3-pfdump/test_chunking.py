"""Exercise the production chunk/context helpers using the host callback adapter."""
from pathlib import Path
import subprocess
import tempfile
import unittest


class ChunkingTest(unittest.TestCase):
    def test_chunking_and_context(self):
        source = r'''
#define main callback_test_main
#include "test_callback.cpp"
#undef main
#include <limits>
int main() {
    int cases = 0;
    for (size_t batch : {2048u, 512u}) {
        for (size_t prompt : {size_t(1), batch-1, batch, batch+1, size_t(3107), 2*batch+5}) {
            size_t offset = 0, chunks = 0;
            while (offset < prompt) {
                size_t count = prompt_chunk_size(prompt-offset, batch);
                assert(count > 0 && count <= batch && offset+count <= prompt);
                assert(count == (prompt-offset < batch ? prompt-offset : batch));
                offset += count;
                ++chunks;
            }
            assert(offset == prompt);
            assert(chunks == (prompt+batch-1)/batch);
            validate_prompt_size(prompt, 2048, prompt+2048); // exact fit
            validate_prompt_size(prompt, 0, prompt);
            bool rejected = false;
            try { validate_prompt_size(prompt, 2048, prompt+2047); }
            catch (const std::runtime_error & e) {
                rejected = std::string(e.what()).find("exceeds ctx-size") != std::string::npos;
            }
            assert(rejected);
            ++cases;
        }
    }
    for (auto values : {std::pair<size_t, int>{65537, 0}, {65536, 1},
                        {std::numeric_limits<size_t>::max(), 1}, {0, 1}, {1, -1}}) {
        bool rejected = false;
        try { validate_prompt_size(values.first, values.second, 65536); }
        catch (const std::runtime_error &) { rejected = true; }
        assert(rejected);
    }
    bool rejected = false;
    try { prompt_chunk_size(1, 0); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    std::cout << "PASS " << cases << " prompt/batch cases; exact-fit, overflow, invalid context/batch checks\n";
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / 'chunking.cpp'
            binary = Path(directory) / 'chunking'
            cpp.write_text(source)
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-I', str(Path(__file__).resolve().parent),
                            str(cpp), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
