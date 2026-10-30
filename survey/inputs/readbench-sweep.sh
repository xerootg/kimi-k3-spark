#!/bin/bash
F=/srv/kimi-k3/store/kimi-k3-ud-iq1_s.experts; B=~/readbench
for h in 1 0; do $B $F pool 1024 16 $h; done           # current engine shape
for c in 256 512 1024 2048 4096 0; do $B $F pool $c 16 1; done
for t in 8 24 32; do $B $F pool 1024 $t 1; done
for c in 128 256 512 1024 2048 4096 0; do for q in 16 32 64 128; do $B $F uring $c $q 1; done; done
$B $F uring 128 128 0
$B $F uring-sqpoll 128 128 1
$B $F uring-sqpoll 1024 32 1
echo "--- single-record fills (k=1) and big fills (k=13)"
$B $F pool 1024 16 1 400 1; $B $F uring 128 64 1 400 1; $B $F uring 256 64 1 400 1
$B $F pool 1024 16 1 200 13; $B $F uring 128 128 1 200 13
echo readbench-done
