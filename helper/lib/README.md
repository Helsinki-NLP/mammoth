# Purpose of this Directory

The symbolic links in this directory ensure that both
`librccl-net-ofi.so` and `librccl-net.so` can be found on the LD_PATH.
Now the situation is that the container provides
`/opt/aws-ofi-rccl/librccl-net.so` but RCCL will look for
`librccl-net-ofi.so`. For some sort of bug in the LUMI software stack
(or the image), RCCL does not fall back to the default filename,
`librccl-net.so`, thus failing to locate the plugin.  There is way to
specify the ofi ending of the plugin, but not its omission.  In my
fix, I have added the current directory to the LD_PATH and ensured
that both filenames are findable.

This readme has also another function.  In 4-module-loads.sh, I check
that the files of this directory are present, but the mentioned
symbolic links are broken outside the container.  Instead, this README
is findable and not a broken link, which satisfies the directory check.
