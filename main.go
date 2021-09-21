// Copyright 2021 The LU Decomposition Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tc128"
)

const (
	// Size is the size of the matrix
	Size = 8
)

// https://www.geeksforgeeks.org/determinant-of-a-matrix/
func cofactor(mat, temp []complex128, p, q, n int) {
	i, j := 0, 0
	for row := 0; row < n; row++ {
		for col := 0; col < n; col++ {
			if row != p && col != q {
				temp[i*Size+j] = mat[row*Size+col]
				j++
				if j == n-1 {
					j = 0
					i++
				}
			}
		}
	}
}

func determinant(mat []complex128, n int) complex128 {
	if n == 1 {
		return mat[0]
	}
	var d complex128
	temp := make([]complex128, Size*Size)
	sign := complex128(1)
	for f := 0; f < n; f++ {
		cofactor(mat, temp, 0, f, n)
		d += sign * mat[f] * determinant(temp, n-1)
		sign = -sign
	}
	return d
}

func main() {
	input := tc128.NewSet()
	input.Add("a", Size, Size)

	set := tc128.NewSet()
	set.Add("l", Size, Size)
	set.Add("u", Size, Size)

	random128 := func(a, b float64) complex128 {
		//return complex((b-a)*rand.Float64()+a, (b-a)*rand.Float64()+a)
		return complex(rand.NormFloat64(), rand.NormFloat64())
	}

	for i := range input.Weights[:1] {
		w := input.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, random128(-1, 1))
			}
		} else {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, random128(-1, 1))
			}
		}
	}

	set.Weights[0].X = set.Weights[0].X[:cap(set.Weights[0].X)]
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			if j == i {
				set.Weights[0].X[i*Size+j] = 1
			} else if j < i {
				set.Weights[0].X[i*Size+j] = random128(-1, 1)
			}
		}
	}

	set.Weights[1].X = set.Weights[1].X[:cap(set.Weights[1].X)]
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			if j >= i {
				set.Weights[1].X[i*Size+j] = random128(-1, 1)
			}
		}
	}

	l1 := tc128.Mul(set.Get("l"), tc128.T(set.Get("u")))
	l2 := tc128.Sub(input.Get("a"), l1)
	cost := tc128.Avg(tc128.Hadamard(l2, l2))

	eta, iterations := complex128(.1+.1i), 256*1024
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := complex128(0)
		set.Zero()

		total += tc128.Gradient(cost).X[0]
		sum := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm := float64(math.Sqrt(float64(sum)))
		scaling := float64(1)
		if norm > 1 {
			scaling = 1 / norm
		}

		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				if j < i {
					set.Weights[0].X[i*Size+j] -= eta * set.Weights[0].D[i*Size+j] * complex(scaling, 0)
				}
			}
		}

		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				if j >= i {
					set.Weights[1].X[i*Size+j] -= eta * set.Weights[1].D[i*Size+j] * complex(scaling, 0)
				}
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: math.Log10(cmplx.Abs(total))})
		fmt.Println(i, cmplx.Abs(total))
		i++
		if cmplx.Abs(total) < 1 {
			break
		}
	}

	d := complex128(1.0)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			if i == j {
				d *= set.Weights[1].X[i*Size+j]
			}
		}
	}
	fmt.Println(d, "=", determinant(set.Weights[1].X, Size))

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}

	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			fmt.Printf("%v ", cmplx.Abs(set.Weights[0].X[i*Size+j]))
		}
		fmt.Println("")
	}
	fmt.Println("")
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			fmt.Printf("%v ", cmplx.Abs(set.Weights[1].X[i*Size+j]))
		}
		fmt.Println("")
	}
}
