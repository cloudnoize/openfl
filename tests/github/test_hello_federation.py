# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import argparse
from pathlib import Path
from subprocess import check_call
from concurrent.futures import ProcessPoolExecutor

from openfl.utilities.utils import rmtree
from tests.github.utils import create_collaborator, create_certified_workspace, certify_aggregator, is_path_name_allowed
from openfl.utilities.utils import getfqdn_env


def main():
    # Test the pipeline
    parser = argparse.ArgumentParser()
    workspace_choice = []
    excluded_dirs = ['workspace', 'default', "experimental"]
    for root, _, files in os.walk('openfl-workspace'):
        if any(file.endswith(".workspace") for file in files):
            dir_path = os.path.relpath(root, 'openfl-workspace')
            dir_path = dir_path.replace(os.sep, '/')
            if dir_path and not any(dir_path.startswith(prefix) for prefix in excluded_dirs):
                workspace_choice.append(dir_path)
    parser.add_argument('--template', default='keras/mnist', choices=workspace_choice)
    parser.add_argument('--fed_workspace', default='fed_work12345alpha81671')
    parser.add_argument('--col1', default='one123dragons')
    parser.add_argument('--col2', default='beta34unicorns')
    parser.add_argument('--rounds-to-train')
    parser.add_argument('--col1-data-path', default='1')
    parser.add_argument('--col2-data-path', default='2')
    parser.add_argument('--save-model')

    origin_dir = Path.cwd().resolve()
    args = parser.parse_args()
    fed_workspace = args.fed_workspace

    # Check if the path name is allowed before creating the workspace
    if not is_path_name_allowed(fed_workspace):
        print(f"The path name {fed_workspace} is not allowed")
        return

    archive_name = f'{fed_workspace}.zip'
    fqdn = getfqdn_env()
    template = args.template
    rounds_to_train = args.rounds_to_train
    col1, col2 = args.col1, args.col2
    col1_data_path, col2_data_path = args.col1_data_path, args.col2_data_path
    save_model = args.save_model

    # START
    # =====
    # Make sure you are in a Python virtual environment with the FL package installed.
    create_certified_workspace(fed_workspace, template, fqdn, rounds_to_train)
    certify_aggregator(fqdn)

    workspace_root = Path().resolve()  # Get the absolute directory path for the workspace

    # Create collaborator #1
    create_collaborator(col1, workspace_root, col1_data_path, archive_name, fed_workspace)

    # Create collaborator #2
    create_collaborator(col2, workspace_root, col2_data_path, archive_name, fed_workspace)

    # Run the federation
    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.submit(check_call, ['fx', 'aggregator', 'start'], cwd=workspace_root)
        time.sleep(5)

        dir1 = workspace_root / col1 / fed_workspace
        executor.submit(check_call, ['fx', 'collaborator', 'start', '-n', col1], cwd=dir1)

        dir2 = workspace_root / col2 / fed_workspace
        executor.submit(check_call, ['fx', 'collaborator', 'start', '-n', col2], cwd=dir2)

    # Convert model to native format
    if save_model:
        check_call(
            ['fx', 'model', 'save', '-i', f'./save/last.pbuf', '-o', save_model],
            cwd=workspace_root)

    os.chdir(origin_dir)
    rmtree(workspace_root)


if __name__ == '__main__':
    main()
