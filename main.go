// Copyright 2021 The LU Decomposition Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	//"math/cmplx"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	//"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"
)

const (
	// Size is the size of the matrix
	Size = 8
)

// https://www.geeksforgeeks.org/determinant-of-a-matrix/
func cofactor(mat, temp []float32, p, q, n int) {
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

func determinant(mat []float32, n int) float32 {
	if n == 1 {
		return mat[0]
	}
	var d float32
	temp := make([]float32, Size*Size)
	sign := float32(1)
	for f := 0; f < n; f++ {
		cofactor(mat, temp, 0, f, n)
		d += sign * mat[f] * determinant(temp, n-1)
		sign = -sign
	}
	return d
}

func main() {
	input := tf32.NewSet()
	input.Add("a", Size, Size)

	set := tf32.NewSet()
	set.Add("vectors", Size, Size)
	set.Add("values", Size, Size)

	random128 := func(a, b float32) float32 {
		//return complex((b-a)*rand.Float64()+a, (b-a)*rand.Float64()+a)
		return float32(rand.NormFloat64())
	}

	for i := range input.Weights {
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

	for i := range set.Weights[:1] {
		w := set.Weights[i]
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

	set.Weights[1].X = set.Weights[1].X[:cap(set.Weights[1].X)]
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			if j == i {
				set.Weights[1].X[i*Size+j] = 1
			}
		}
	}

	l1 := tf32.Mul(input.Get("a"), set.Get("vectors"))
	l2 := tf32.Mul(tf32.T(set.Get("vectors")), set.Get("values"))
	l3 := tf32.Sub(l1, l2)
	cost := tf32.Avg(tf32.Hadamard(l3, l3))

	eta, iterations := float32(1), 256*1024
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := float32(0.0)
		set.Zero()

		total += tf32.Gradient(cost).X[0]
		sum := float32(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := float32(math.Sqrt(float64(sum)))
		scaling := float32(1)
		if norm > 1 {
			scaling = 1 / norm
		}

		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				set.Weights[0].X[i*Size+j] -= eta * set.Weights[0].D[i*Size+j] * scaling
			}
		}

		l := 0
		for k := 0; k < Size; k++ {
			for j := 0; j < Size; j++ {
				if j == k && i/1000 <= l {
					set.Weights[1].X[k*Size+j] -= eta * set.Weights[1].D[k*Size+j] * scaling
				}
				if j == k {
					l++
				}
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: math.Log10(float64(total))})

		d := float32(1.0)
		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				if i == j {
					d *= set.Weights[1].X[i*Size+j]
				}
			}
		}

		fmt.Println(i, total, d)
		i++
		if total < 5e-12 {
			break
		}
	}

	d := float32(1.0)
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			if i == j {
				d *= set.Weights[1].X[i*Size+j]
			}
		}
	}
	fmt.Println(d, "=", determinant(input.Weights[0].X, Size))

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
			fmt.Printf("%v ", set.Weights[1].X[i*Size+j])
		}
		fmt.Println("")
	}
	fmt.Println("")
	l3(func(a *tf32.V) bool {
		for i := 0; i < Size; i++ {
			for j := 0; j < Size; j++ {
				fmt.Printf("%v ", a.X[i*Size+j])
			}
			fmt.Println("")
		}
		return true
	})
}
