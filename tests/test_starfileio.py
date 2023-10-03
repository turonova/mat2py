import unittest
import starfileio as star


class TestReader(unittest.TestCase):
    def test(self):
        for name in ["run_data.star", "run_data_no_optics.star", "stopgap_particles.star", "stopgap_wedge.star"]:

            frames, specifiers, comments = star.read(f"../example_files/{name}")

            # TODO: the comments from the reader's and the writer's input are slightly different
            star.write(frames, f"./outputs/{name}", specifiers, list(map(lambda lst: None if lst is None or len(lst) == 0 else lst[0], comments)), True)
