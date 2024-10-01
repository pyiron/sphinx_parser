import unittest
from stinx.source import generator
import yaml
import os


class TestStinx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        cls.file_name = cls.current_dir + "/../../stinx/source/input_data.yml"

    def test_stinx(self):
        with open(self.file_name, "r") as f:
            file_content = f.read()
        all_data = yaml.safe_load(file_content)
        all_data = generator.replace_alias(all_data)
        all_files = "\n\n".join(generator.get_all_functions(all_data))
        self.assertTrue("def get_all" in all_files)


if __name__ == '__main__':
    unittest.main()
