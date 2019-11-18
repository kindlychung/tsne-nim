import arraymancer
import math, parsecsv, strutils, sequtils, strformat, sugar, logging

proc hbeta*[T](d: Tensor[T], beta: float=1.0): auto =
  var
    p = (d * beta).map(x => exp(-x))
  let
    sumP = sum(p, 0)
    h = sumP.map(x => ln(x)) .+ (beta * (sum(d .* p) ./ sumP))
  p ./= sumP
  return (h, p)

proc get_row_without_diagonal[T](t:Tensor[T], r:int):Tensor[T] =
    let
        s = t.shape[0] - 1
    if r == 0: t[r+1..s]
    elif r == s: t[0..<s]
    else:
      concat(t[0..<r], t[r+1..s], axis = 0)

proc set_row_without_diagonal[T](t:var Tensor[T], r:int, input:Tensor[T]):Tensor[T] =
    result = t
    let
        s = t.shape[0] - 1
    if r == 0: t[r, r+1..s] = input.unsqueeze(0)
    elif r == s: t[r, 0..<s] = input.unsqueeze(0)
    else:
        t[r, 0..<r] = input[0..<r].unsqueeze(0)
        t[r, r+1..s] = input[r..<input.shape[0]].unsqueeze(0)

proc x2p[T](x: Tensor[T], tol: float=1e-5, perplexity: float=30.0):auto =
  info "Computing pairwise distances..."
  let
      n = x.shape[0]
      sum_x = sum(square(x), 1)
      D = (((x * x.transpose) * (-2.0)) .+ sum_x).transpose .+ sum_x
      logU = ln(perplexity)
  var
      P = zeros[T](n,n)
      beta = ones[T](n)
  for i in 0..<n:
    if i %% 500 == 0:
      info &"Computing P-values for point {i} of {n}..."
    # Compute the Gaussian kernel and entropy for the current precision
    var
        betamin = -Inf
        betamax = Inf
        Di = get_row_without_diagonal[T](D[i,_].squeeze(), i)
        (H, thisP) = hbeta[float](Di, beta[i])
        Hdiff = H .- logU
        tries = 0
    while abs(Hdiff).map_inline(x > tol).reduce((a,b:bool) => a or b) and tries < 50:
      if (Hdiff .> zeros[float](Hdiff.shape)).reduce((a,b:bool) => a or b):
        betamin = beta[i]
        if betamax == Inf or betamax == -Inf:
            beta[i] = beta[i] * 2.0
        else:
            beta[i] = (beta[i] + betamax) / 2.0
      else:
        betamax = beta[i]
        if betamin == Inf or betamin == -Inf:
            beta[i] = beta[i].float / 2.0
        else:
            beta[i] = (beta[i] + betamin) / 2.0

      # Recompute the values
      (H, thisP) = hbeta[float](Di, beta[i])
      Hdiff = H .- logU
      tries += 1
    P = set_row_without_diagonal[T](P, i, thisP)
  let
      sig = mean(beta.map((b: float) => sqrt(1.0 / b)))
  info &"Mean value of sigma : {sig}"
  return P

proc maximum[T:SomeFloat](v: T):auto =
    if v.classify == fcNaN:
        info &"{v} is NaN"
        v
    else:
        max(v, 1e-12)

proc loadTensor[T:SomeFloat](fn: string): Tensor[T] =
  var
      p: CsvParser
      line = ""
      rows: seq[seq[float]] = @[]
  p.open(fn)
  while p.readRow():
      var row: seq[float] = @[]
      for val in items(p.row):
          row.add(parseFloat(val))
      rows.add(row)
  close(p)
  rows.toTensor.astype(T)

proc gPlus2[T](g: Tensor[T], b: Tensor[bool]): Tensor[T] =
    result = g.clone()
    for g, comp in mzip(result, b):
        if comp:
            g += 0.2
        else:
            g = 0.0

proc gMultiple8[T](g: Tensor[T], b: Tensor[bool]): Tensor[T] =
    result = g.clone()
    for g, comp in mzip(result, b):
        if comp:
            g *= 0.8
        else:
            g = 0.0

proc tsne*[T:SomeFloat](xi: Tensor[T], dims: int=2, initial_dims: int=50, perplexity: float=30.0):auto =
  const
      max_iter = 500
      initial_momentum = 0.5
      final_momentum = 0.8
      eta = 500.float
      min_gain = 0.01

  let
      X = pca(xi, initial_dims).projected
      n = X.shape[0]
      zero = zeros[T](n, dims)
  var
      Y = randomNormalTensor[T]([n, dims])
      dY = zeros[T](n, dims)
      iY = zeros[T](n, dims)
      gains = ones[T](n, dims)
      P = x2p[T](X, 1e-5, perplexity)
  P = P .+ transpose(P)
  P = (P / sum(P) * 4.0).map(maximum[T])
  for i in 0..<max_iter:
      let 
          sumY = sum(square(Y), 1)
          num = -2.0 * (Y * Y.transpose)
      var
          numT = 1.0 ./ (1.0 .+ (transpose(num .+ sumY) .+ sumY))
      for j in 0..<n:
        numT[j, j] = 0.0
      let
          Q = (numT / sum(numT)).map(maximum[T])
          PQ = P - Q
      for j in 0..<n:
          let
              tt = sum((PQ[_, j] .* numT[_, j]) .* (Y[j, _] .- Y), 0)
          dY[j,_] = tt

      let
          momentum = if i < 20:
                       initial_momentum
                     else:
                       final_momentum
          dyz = dY .> zero
          iyz = iY .> zero
      gains = gPlus2[float](gains, (dyz .!= iyz)) + gMultiple8[float](gains, (dyz .== iyz))

      proc compare_min_gain[T](x: T):T =
          result = x
          if x < min_gain:
              result = min_gain
      gains.apply(compare_min_gain)
      iY = momentum * iY - (eta * (gains .* dY))
      Y .+= iY
      Y .-= mean(Y, axis=0)

      if (i + 1) %% 10 == 0:
        let
            C = sum(P .* ln(P ./ Q))
        info &"Iteration {i + 1}: error is {C}"

      if i == 100:
        P ./= 4.0
  return Y

when isMainModule:
   var l = newConsoleLogger()
   addHandler(l)
   var
       p: CsvParser
       line = ""
       rows: seq[seq[float]] = @[]
   p.open("fixture/mnist2500_X.txt")
   while p.readRow():
      var row: seq[float] = @[]
      for val in items(p.row):
        row.add(parseFloat(val))
      rows.add(row)
   p.close
   var
       label: seq[int] = @[]
   p.open("fixture/mnist2500_labels.txt")
   while p.readRow:
       for val in items(p.row):
         label.add(int(parseFloat(val)))
   p.close

   let
       P: Tensor[float] = tsne[float](rows.toTensor, 2, 50, 20.0)
       #p1 = tsne[float](rows.toTensor)
   var f = open("out.csv", fmWrite)
   for i in 0..<P.shape[0]:
     f.writeLine(&"{P[i,0]},{P[i,1]}")
   f.close

   info &"{P[0,_]}, {P.shape}"
