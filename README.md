# IT4851 - Nhom 13

## Installation

## Usage

## Deployment Guideline

1. Jupyterlab: http://23.98.73.116:8888/
2. Deploy server:
    1. Stop old instance:
        - `tmux list-sessions`. Find the lastest session in list. For example:
            ```
            0: 1 windows (created Sun Jun 14 08:11:52 2020) [110x46] (attached)
            3: 1 windows (created Tue Jun 16 01:18:56 2020) [110x54] (attached)
            5: 1 windows (created Wed Jun 17 09:21:33 2020) [80x23]
            ```
        - `tmux attach-session -t <Number>`. For example: `tmux attach-session -t 5`
        - Press `Ctrl + C` to stop instance, type `exit` to exit session.
    2. Pull lastest code:
        ```bash
        cd it4851_nhom13
        git pull
        ```
    3. Start new instance:
        - Run `tmux`
        - Run `flask run`. Press `Ctrl + B`, then `D`
