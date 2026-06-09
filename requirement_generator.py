#!/usr/bin/env python

import sys
import pkg_resources

def print_main_library_versions(libraries):
    """
    print the version for the specified libraries if available.
    skip if not installed or doesn't have a version.
    """
    print("Checking versions of important libraries:\n")
    for lib in libraries:
        try:
            dist = pkg_resources.get_distribution(lib)
            print(f"{lib} version: {dist.version}")
        except pkg_resources.DistributionNotFound:
            print(f"{lib} is not installed or not found.")
        except Exception as e:
            print(f"Could not get version for {lib}: {e}")
    print("\nDone checking specified libraries.\n")


def generate_minimal_requirements(libraries, output_file='requirements.txt'):
    """
    generate a minimal requirements file for the selected libraries in your environment.
    """
    print(f"Generating a minimal requirements file: {output_file}")
    req_lines = []
    for lib in libraries:
        try:
            dist = pkg_resources.get_distribution(lib)
            req_lines.append(f"{dist.project_name}>={dist.version}")
        except pkg_resources.DistributionNotFound:
            pass  # Skip if not found
    with open(output_file, 'w') as f:
        for line in req_lines:
            f.write(line + "\n")
    print(f"Requirements have been written to {output_file}.")


if __name__ == "__main__":
    # can be edit the list with main/important libraries
    important_libraries = [
        'torch',
        'numpy',
        'pandas',
        'simpy',
    ]

    print("Python version:", sys.version)
    print_main_library_versions(important_libraries)
    generate_minimal_requirements(important_libraries, 'requirements.txt')
