# Introduction
This repo contains code for our group project

# Step to request GPU in PACE
1. Login to PACE with: `ssh <GT ID>@gatech.pace.gatech.edu` - replace `<GT ID>` with your GT ID
2. `cd` to the project root dir
3. Open a `tmux` session, if you want to run the training on the background (optional, but highly recommended)
    - You may need to look up `tmux` tutorial if you are not familiar with it
4. `salloc --gres=gpu:H100:1 --ntasks-per-node=1 --time 03:00:00` - submit a request for one node with `H100` GPU and one task per node
    - adjust the request time (in hours) for your need
    - there are different GPUs you can request, look up PACE documentation for more details