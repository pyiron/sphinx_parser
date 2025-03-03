import unittest
from sphinx_parser.src import generator
import yaml
import os


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        cls.file_name = cls.current_dir + "/../../sphinx_parser/src/input_data.yml"

    def test_sphinx_parser(self):
        with open(self.file_name, "r") as f:
            file_content = f.read()
        all_data = yaml.safe_load(file_content)
        all_data = generator._replace_alias(all_data)
        self.assertTrue("def create" in generator._get_class(all_data))

    def test_all_content(self):
        with open(self.current_dir + "/../../sphinx_parser/input.py", "r") as f:
            file_content = f.read()
        self.assertEqual(file_content, generator._get_file_content())


if __name__ == "__main__":
    unittest.main()
