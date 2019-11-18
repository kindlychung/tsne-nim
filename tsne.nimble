# Package

version       = "0.0.1.0"
author        = "tomohiko okazaki"
description   = "TSNE example"
license       = "MIT"
srcDir        = "src"
installExt    = @["bin"]
bin           = @["tsne"]
binDir        = "result/bin"

# Dependencies

requires "nim >= 1.0.0"
requires "nimblas 0.2.2"
requires "Arraymancer"

