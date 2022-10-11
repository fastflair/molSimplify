from molSimplify.Scripts.generator import startgen
from pkg_resources import resource_filename, Requirement

infile = resource_filename(Requirement.parse(
    "molSimplify"), "tests/inputs/example_1_noff.in")
args = ['main.py', '-i', infile]
startgen(args, False, False)
infile = resource_filename(Requirement.parse(
    "molSimplify"), "tests/inputs/example_1.in")
args = ['main.py', '-i', infile]
startgen(args, False, False)
