/*============================================================================*/
/*
  VFLib: https://github.com/vinniefalco/VFLib

  Copyright (C) 2008 by Vinnie Falco <vinnie.falco@gmail.com>

  This library contains portions of other open source products covered by
  separate licenses. Please see the corresponding source files for specific
  terms.
  
  VFLib is provided under the terms of The MIT License (MIT):

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/
/*============================================================================*/

#ifndef VF_DISTANCETRANSFORM_VFHEADER
#define VF_DISTANCETRANSFORM_VFHEADER

#include <opencv2/opencv.hpp>

#include "Util.h"

namespace vf
{

/** Distance transform calculations.

    @ingroup vf_gui
*/
struct DistanceTransform
{
  //----------------------------------------------------------------------------

  /** Mask inclusion test functor.
  */
  struct WhiteTest
  {
    explicit WhiteTest (cv::Mat src) : m_src (src)
    {
    }

    inline bool operator () (int const x, int const y) const noexcept
    {
      uint8_t val = m_src.at<uint8_t>(y, x);
      return val < 128;
    }

  private:
    cv::Mat m_src;
  };

  /** Mask exclusion test functor.
  */
  struct BlackTest
  {
    explicit BlackTest (cv::Mat src) : m_src (src)
    {
    }

    inline bool operator () (int const x, int const y) const noexcept
    {
      uint8_t val = m_src.at<uint8_t>(y, x);
      return val > 127;
    }

  private:
    cv::Mat m_src;
  };

  /** Mask presence test functor.
  */
  struct AlphaTest
  {
    explicit AlphaTest (cv::Mat src) : m_src (src)
    {
    }

    inline bool operator () (int const x, int const y) const noexcept
    {
      uint8_t val = m_src.at<uint8_t>(y, x);
      return val != 0;
    }

  private:
    cv::Mat m_src;
  };

  //------------------------------------------------------------------------------

  // Distance output to 8-bit unsigned.
  //
  // The radius parameter provides a way to set a maximum radius
  // for distance values generated during the calculation. The
  // radius should be the largest value possible in the distance
  // calculation since the computed distance is divided by the radius
  // to get a normalized distance value. For example, a 5 x 10 rectange
  // would have a max radius of sqrt(5/2*5/2 + 10/2*10/2)+0.5
  // This scale protects against floating point overflow.

  struct OutputDistancePixels
  {
    OutputDistancePixels (cv::Mat dest, int radius)
      : m_dest (dest)
      , m_radius (radius)
      , m_radiusSquared (radius * radius)
    {
      // Check for overflow in (radius * radius)
      assert(m_radiusSquared > m_radius);
    }

    void operator () (int const x, int const y, double distance)
    {      
      if (distance <= m_radiusSquared && distance > 0)
      {
        double scaledDistance;
        uint8_t byteVal;
        
        if (distance == 1) {
          scaledDistance = 1;
          byteVal = 1;
        } else {
          scaledDistance = sqrt(distance) / m_radius;
          byteVal = uint8_t((scaledDistance * 255) + 0.5);
          
          // Lower limit for scaled distance is 1, but the sqrt() / rad
          // operation could round down to zero.
          
          if (byteVal == 0) {
            byteVal = 1;
          }
        }
        
        if (distance > 0 && (byteVal == 0)) {
          // Scaling the value by distance has reduced a non-zero
          // value down to zero. Not good.
          assert(0);
        }
        
        m_dest.at<uint8_t>(y, x) = byteVal;
      }
      else
      {
        uint8_t val = 0;
        m_dest.at<uint8_t>(y, x) = val;
      }
    }

  private:
    cv::Mat m_dest;
    int m_radius;
    int m_radiusSquared;
  };

  //----------------------------------------------------------------------------
  // 
  // "A General Algorithm for Computing Distance Transforms in Linear Time"
  //  - A. Meijster, 2003
  //
  struct Meijster
  {
    struct EuclideanMetric
    {
      template <class T>
      static inline T f (T x_i, T gi) noexcept
      {
        return (x_i*x_i)+(gi*gi);
      }

      template <class T>
      static inline T sep (T i, T u, T gi, T gu, T) noexcept
      {
        return (u*u - i*i + gu*gu - gi*gi) / (2*(u-i));
      }
    };

    //--------------------------------------------------------------------------

    struct ManhattanMetric
    {
      template <class T>
      static inline T f (T x_i, T gi) noexcept
      {
        return abs (x_i) + gi;
      }

      template <class T>
      static inline int sep (T i, T u, T gi, T gu, T inf) noexcept
      {
        T const u_i = u - i;

        if (gu >= gi + u_i)
          return inf;
        else if (gi > gu + u_i)
          return -inf;
        else
          return (gu - gi + u + i) / 2;
      }
    };

    //--------------------------------------------------------------------------

    struct ChessMetric
    {
      template <class T>
      static inline T f (T x_i, T gi) noexcept
      {
        return maxi(abs (x_i), gi);
      }

      template <class T>
      static inline int sep (T i, T u, T gi, T gu, T) noexcept
      {
        if (gi <= gu)
          return maxi(i+gu, (i+u)/2);
        else
          return mini(u-gi, (i+u)/2);
      }
    };

    //--------------------------------------------------------------------------

    template <class Functor, class BoolImage, class Metric>
    static void calculate (Functor f, BoolImage test, int const m, int const n, Metric metric)
    {
      std::vector <int> g (m * n);

      int const inf = m + n;

      // phase 1
      {
        for (int x = 0; x < m; ++x)
        {
          g [x] = test (x, 0) ? 0 : inf;

          // scan 1
          for (int y = 1; y < n; ++y)
          {
            int const ym = y*m;
            g [x+ym] = test (x, y) ? 0 : 1 + g [x+ym-m];
          }

          // scan 2
          for (int y = n-2; y >=0; --y)
          {
            int const ym = y*m;

            if (g [x+ym+m] < g [x+ym])
              g [x+ym] = 1 + g[x+ym+m];
          }
        }
      }

      // phase 2
      {
        std::vector <int> s (maxi(m, n));
        std::vector <int> t (maxi(m, n));

        for (int y = 0; y < n; ++y)
        {
          int q = 0;
          s [0] = 0;
          t [0] = 0;

          int const ym = y*m;

          // scan 3
          for (int u = 1; u < m; ++u)
          {
            while (q >= 0 && metric.f (t[q]-s[q], g[s[q]+ym]) > metric.f (t[q]-u, g[u+ym]))
              q--;

            if (q < 0)
            {
              q = 0;
              s [0] = u;
            }
            else
            {
              int const w = 1 + metric.sep (s[q], u, g[s[q]+ym], g[u+ym], inf);

              if (w < m)
              {
                ++q;
                s[q] = u;
                t[q] = w;
              }
            }
          }

          // scan 4
          for (int u = m-1; u >= 0; --u)
          {
            int const d = metric.f (u-s[q], g[s[q]+ym]);
            f (u, y, d);
            if (u == t[q])
              --q;
          }
        }
      }
    }

    //--------------------------------------------------------------------------

    template <class T>
    static T floor_fixed8 (T x)
    {
      return x & (~T(0xff));
    }

    // Mask: 0 = point of interest
    //
    template <class Functor, class Mask, class Metric>
    static void calculateAntiAliased (Functor f, Mask mask, int const m, int const n, Metric metric)
    {
      int64 const scale = 256;

      std::vector <int64> g (m * n);

      int64 const inf = scale * (m + n);

      // phase 1
      {
        for (int x = 0; x < m; ++x)
        {
          int a;

          a = mask (x, 0);

          if (a == 0)
            g [x] = 0;
          else if (a == 255)
            g [x] = inf;
          else
            g [x] = a;

          // scan 1
          for (int y = 1; y < n; ++y)
          {
            int const idx = x+y*m;

            a = mask (x, y);
            if (a == 0)
              g [idx] = 0;
            else if (a == 255)
              g [idx] = scale + g [idx-m];
            else
              g [idx] = a;
          }

          // scan 2
          for (int y = n-2; y >=0; --y)
          {
            int const idx = x+y*m;
            int64 const d = scale + g [idx+m];
            if (g [idx] > d)
              g [idx] = d;
          }
        }
      }

      // phase 2
      {
        std::vector <int> s (maxi(m, n));
        std::vector <int64> t (maxi(m, n)); // scaled

        for (int y = 0; y < n; ++y)
        {
          int q = 0;
          s [0] = 0;
          t [0] = 0;

          int const ym = y*m;

          // scan 3
          for (int u = 1; u < m; ++u)
          {
            while (q >= 0 && metric.f (floor_fixed8(t[q]) - scale*s[q], g[s[q]+ym]) >
                             metric.f (floor_fixed8(t[q]) - scale*u, g[u+ym]))
            {
              q--;
            }

            if (q < 0)
            {
              q = 0;
              s [0] = u;
            }
            else
            {
              int64 const w = scale + metric.sep (scale*s[q], scale*u, g[s[q]+ym], g[u+ym], inf);

              if (w < scale * m)
              {
                ++q;
                s[q] = u;
                t[q] = w;
              }
            }
          }

          // scan 4
          for (int u = m-1; u >= 0; --u)
          {
            int64 const d = metric.f (scale*(u-s[q]), g[s[q]+ym]);
            f (u, y, d);
            if (u == t[q]/scale)
              --q;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    
    template <class Mask>
    struct Phase1
    {
      Phase1 (int m_, int n_, int64* g_, Mask mask_)
        : m (m_), n (n_), g (g_), mask (mask_)
      {
      }

      inline void operator() (int x) noexcept
      {
        int64 const inf = 256 * (m + n);
  
        int a;

        a = mask (x, 0);

        if (a == 0)
          g [x] = 0;
        else if (a == 255)
          g [x] = inf;
        else
          g [x] = a;

        // scan 1
        for (int y = 1; y < n; ++y)
        {
          int const idx = x+y*m;

          a = mask (x, y);
          if (a == 0)
            g [idx] = 0;
          else if (a == 255)
            g [idx] = 256 + g [idx-m];
          else
            g [idx] = a;
        }

        // scan 2
        for (int y = n-2; y >=0; --y)
        {
          int const idx = x+y*m;
          int64 const d = 256 + g [idx+m];
          if (g [idx] > d)
            g [idx] = d;
        }
      }
    
    private:
      int m;
      int n;
      int64* g;
      Mask mask;
    };

    template <class Functor, class Mask, class Metric>
    static void calculateAntiAliasedLoop (
      Functor f, Mask mask, int const m, int const n, Metric metric)
    {
      int64 const scale = 256;

      std::vector <int64> g (m * n);

      int64 const inf = scale * (m + n);

      // phase 1
      {
        Phase1 <Mask> p (m, n, &g[0], mask);
        for (int x = 0; x < m; ++x)
          p (x);
        //loop.operator() <Phase1 <Mask> > (m, m, n, &g[0], mask);

#if 0
        for (int x = 0; x < m; ++x)
        {
          int a;

          a = mask (x, 0);

          if (a == 0)
            g [x] = 0;
          else if (a == 255)
            g [x] = inf;
          else
            g [x] = a;

          // scan 1
          for (int y = 1; y < n; ++y)
          {
            int const idx = x+y*m;

            a = mask (x, y);
            if (a == 0)
              g [idx] = 0;
            else if (a == 255)
              g [idx] = scale + g [idx-m];
            else
              g [idx] = a;
          }

          // scan 2
          for (int y = n-2; y >=0; --y)
          {
            int const idx = x+y*m;
            int64 const d = scale + g [idx+m];
            if (g [idx] > d)
              g [idx] = d;
          }
        }
#endif
      }

      // phase 2
      {
        std::vector <int> s (maxi(m, n));
        std::vector <int64> t (maxi(m, n)); // scaled

        for (int y = 0; y < n; ++y)
        {
          int q = 0;
          s [0] = 0;
          t [0] = 0;

          int const ym = y*m;

          // scan 3
          for (int u = 1; u < m; ++u)
          {
            while (q >= 0 && metric.f (floor_fixed8(t[q]) - scale*s[q], g[s[q]+ym]) >
                             metric.f (floor_fixed8(t[q]) - scale*u, g[u+ym]))
            {
              q--;
            }

            if (q < 0)
            {
              q = 0;
              s [0] = u;
            }
            else
            {
              int64 const w = scale + metric.sep (scale*s[q], scale*u, g[s[q]+ym], g[u+ym], inf);

              if (w < scale * m)
              {
                ++q;
                s[q] = u;
                t[q] = w;
              }
            }
          }

          // scan 4
          for (int u = m-1; u >= 0; --u)
          {
            int64 const d = metric.f (scale*(u-s[q]), g[s[q]+ym]);
            f (u, y, d);
            if (u == t[q]/scale)
              --q;
          }
        }
      }
    }
  };

};

}
  
#endif
