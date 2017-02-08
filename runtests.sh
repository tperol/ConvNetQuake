#!/bin/sh

# Requires pytest package
py.test --cov=quakenet `dirname ${0}`/quakenet
