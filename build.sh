#!/bin/bash

CWD="$(pwd)"
DRY_INSTALL_DIR="${CWD}/dry/install"

cd dry
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

if [ -f $DRY_INSTALL_DIR/setup.sh ]; then
    source $DRY_INSTALL_DIR/setup.sh
    for f in ./dry/install/*; do
	if [ -d "$f" ]; then
	    PACKAGE_DIR="$( basename "$f" )"
	    package="$DRY_INSTALL_DIR/$PACKAGE_DIR"
	    eval 'export PYTHONPATH=$package:$PYTHONPATH'
	    eval 'export PATH=$package:$PATH'
	fi
    done
fi

cd ../wet
# colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release 
export MAKEFLAGS="-j1"
echo "meow !!!"
# colcon build --parallel-workers 1 --symlink-install --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release 
colcon build --parallel-workers 1 --symlink-install --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_PARALLEL_LEVEL=1
