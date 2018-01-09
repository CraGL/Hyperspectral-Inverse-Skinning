#!/usr/bin/env /usr/local/bin/python3

from __future__ import print_function, division

from shlex import quote
import subprocess

solver_parameters = [
	("biquadratic_weighted", "--energy biquadratic --strategy ssv:weighted"),
	...
	]
starting_guesses = [
	("recovery", "--recovery")
	]

for solver_name, solver_param in solver_parameters:
	for start_name, start_param in starting_guesses:
		output_path = "timings/" + solver_name + "+" + start_name + ".out"
		# cmd = "python flat_intersect.py --GT %(GT)s | tee %(output_path)s" % { 'GT': GT, 'output_path': output_path }
		cmd = "ls -l %(GT)s | tee %(output_path)s" % { 'GT': quote(GT), 'output_path': output_path }
		subprocess.call( cmd, shell=True )
