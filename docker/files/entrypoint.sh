#!/bin/bash

# Run SSH server
#/etc/init.d/ssh start

# Run the CMD specified by the user
sh -c "$(echo $@)"
