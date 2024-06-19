/* **************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <ostream>
#include <stdexcept>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <rocblas/rocblas.h>

// If USE_ROCBLAS_REALLOC_ON_DEMAND is false, automatic reallocation is disable and we will manually
// reallocate workspace
#define USE_ROCBLAS_REALLOC_ON_DEMAND true

#ifdef ROCSOLVER_CLIENTS_TEST
#define ROCSOLVER_TEST_CHECK(T, max_error, tol)                                                       \
    {                                                                                                 \
        ASSERT_LE((max_error), (tol)*get_epsilon<T>());                                               \
        std::cout << "[          ] " << "Error / (K * n * ulp) <= "  << (                             \
                (tol > get_safemin<T>()) ? max_error/(tol * get_epsilon<T>()) : get_safemin<T>()      \
        ) << " [number K is test dependent]" << std::endl << std::flush;                              \
    }                                                                                                 \

#else // ROCSOLVER_CLIENTS_BENCH
#define ROCSOLVER_TEST_CHECK(T, max_error, tol)
#endif

typedef enum rocsolver_inform_type_
{
    inform_quick_return,
    inform_invalid_size,
    inform_invalid_args,
    inform_mem_query,
} rocsolver_inform_type;

inline void rocsolver_bench_inform(rocsolver_inform_type it, size_t arg = 0)
{
    switch(it)
    {
    case inform_quick_return: fmt::print("Quick return...\n"); break;
    case inform_invalid_size: fmt::print("Invalid size arguments...\n"); break;
    case inform_invalid_args: fmt::print("Invalid value in arguments...\n"); break;
    case inform_mem_query: fmt::print("{} bytes of device memory are required...\n", arg); break;
    }
    fmt::print("No performance data to collect.\n");
    fmt::print("No computations to verify.\n");
    std::fflush(stdout);
}

// recursive format function (base case)
inline void format_bench_table(std::string&) {}

// recursive format function
template <typename T, typename... Ts>
inline void format_bench_table(std::string& str, T arg, Ts... args)
{
    str += fmt::format("{:<15}", arg);
    if(sizeof...(Ts) > 0)
        str += ' ';
    format_bench_table(str, args...);
}

template <typename... Ts>
void rocsolver_bench_output(Ts... args)
{
    std::string table_row;
    format_bench_table(table_row, args...);
    std::puts(table_row.c_str());
    std::fflush(stdout);
}

inline void rocsolver_bench_header(const char* title)
{
    fmt::print("\n{:=<44}\n{}\n{:=<44}\n", "", title, "");
}

inline void rocsolver_bench_endl()
{
    std::putc('\n', stdout);
    std::fflush(stdout);
}

template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
inline T sconj(T scalar)
{
    return scalar;
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
inline T sconj(T scalar)
{
    return std::conj(scalar);
}

// A struct implicity convertable to and from char, used so we can customize Google Test
// output for LAPACK char arguments without affecting the default char output.
class printable_char
{
    char value;

public:
    printable_char(char c)
        : value(c)
    {
        if(c < 0x20 || c >= 0x7F)
            throw std::invalid_argument(fmt::format(
                "printable_char must be a printable ASCII character (received {:#x})", c));
    }

    operator char() const
    {
        return value;
    }
};

// gtest printers

inline std::ostream& operator<<(std::ostream& os, rocblas_status x)
{
    return os << rocblas_status_to_string(x);
}

inline std::ostream& operator<<(std::ostream& os, printable_char x)
{
    return os << char(x);
}

// location of the sparse data directory for the re-factorization tests
fs::path get_sparse_data_dir();

// Hash arrays following the spirit of `boost::hash_combine`.
template<typename T>
std::size_t hash_combine(std::size_t seed, T value)
{
    using S = decltype(std::real(T{}));
    auto hasher = std::hash<S>();

    if constexpr (rocblas_is_complex<T>)
    {
        seed ^= hasher(std::real(value)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(std::imag(value)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    else
    {
        seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    return seed;
}

template<typename T>
std::size_t hash_combine(std::size_t seed, T const *array, std::size_t array_size)
{
    std::size_t hash = hash_combine(seed, array_size);
    for (std::size_t i = 0; i < array_size; ++i)
    {
        hash = hash_combine(hash, array[i]);
    }

    return hash;
}

template<typename T>
std::size_t hash_combine(std::size_t seed, const std::vector<T>& array)
{
    return hash_combine(seed, array.data(), array.size());
}

/* #define ROCSOLVER_DETAIL_EIGENSOLVERS_USE_NEW_TESTS */

/* #define ROCSOLVER_EIGENSOLVERS_USE_ALTERNATIVE_TESTS_INPUTS */
/* #define ROCSOLVER_EIGENSOLVERS_USE_ALTERNATIVE_TESTS_DETAILS */

template<typename T, typename I = int64_t>
class largest_common_subsequence {
    public:
        using S = decltype(std::real(T{}));

        [[maybe_unused]] auto operator()(T *a, std::size_t size_a, T *b, std::size_t size_b, S tol = 1E-4)
        {
            sseqs_size_= {};
            distance_ = std::numeric_limits<T>::infinity();
            size_a_ = {};
            size_b_ = {};
            sseq_a_ = {};
            sseq_b_ = {};
            pairs_distances_ = {};
            tol_= S(0);

            /* if ((size_a == 0) || (size_b == 0)) */
            /* { */

            /*     start_a_ = 0; */
            /*     start_b_ = 0; */
            /* } */
            /* if (size_a == size_b) */
            /* { */
            /*     start_a_ = 0; */
            /*     start_b_ = 0; */

            /* } */
            /* else if (size_a < size_b) */
            /* { */
            /*     std::size_t si = 0, num = size_b - size_a + 1; */
            /*     T dist = std::numeric_limits<T>::infinity(); */
            /*     for (std::size_t i = 0; i < num; ++i) */
            /*     { */
            /*         T d_ = std::abs(b[i] - a[0]); */
            /*         if (dist > d_) */
            /*         { */
            /*             si = i; */
            /*             dist = d_; */
            /*         } */
            /*     } */
            /*     start_a_ = 0; */
            /*     start_b_ = si; */
            /* } */
            /* else if (size_b < size_a) */
            /* { */
            /*     std::size_t si = 0, num = size_a - size_b + 1; */
            /*     T dist = std::numeric_limits<T>::infinity(); */
            /*     for (std::size_t i = 0; i < num; ++i) */
            /*     { */
            /*         T d_ = std::abs(a[i] - b[0]); */
            /*         if (dist > d_) */
            /*         { */
            /*             si = i; */
            /*             dist = d_; */
            /*         } */
            /*     } */
            /*     start_a_ = si; */
            /*     start_b_ = 0; */
            /* } */
            /* else */
            /* { */
            /*     fmt::print(stderr, "Impossible case at {}:{}\n", __FILE__, __LINE__); */
            /*     rocblas_abort(); */
            /* } */

            memo_distances_.resize(size_a * size_b, S(-1));
            memo_sizes_.resize(size_a * size_b, I(-1));

            sseqs_size_ = std::min(size_a, size_b);
            sseq_a_.resize(sseqs_size_);
            sseq_b_.resize(sseqs_size_);
            distance_ = (sseqs_size_ > 0) ? T(0) : std::numeric_limits<T>::infinity();
            for (size_t i = 0; i < sseqs_size_; ++i)
            {
                T ai = a[start_a_ + i];
                T bi = b[start_b_ + i];
                T d_ = std::abs(ai - bi);
                sseq_a_[i] = ai;
                sseq_b_[i] = bi;
                if (distance_ < d_)
                {
                    distance_ = d_;
                }
                pairs_distances_.insert({d_, i});
            }

            /* start_a = start_a_; */
            /* start_b = start_b_; */

            print_debug();

            return std::make_pair(sseqs_size_, distance_);
        }

        T distance()
        {
            return distance_;
        }

        auto sub_sequences() -> std::pair<std::vector<S>, std::vector<S>>
        {
            return std::make_pair(sseq_a_, sseq_b_);
        }

        auto sub_sequences_size() -> std::size_t
        {
            return sseqs_size_;
        }

        void print_debug(std::size_t num_pairs = 0)
        {
            std::cout << "::: Max distance: " << distance_;
            std::cout << ", subsequence pairs (largest to smallest distance): (";
            num_pairs = (num_pairs == 0) ? pairs_distances_.size() : num_pairs;
            int numel = static_cast<int>(std::min(num_pairs, pairs_distances_.size()));
            for (auto& [_, i] : pairs_distances_)
            {
                numel--;
                std::cout << ""
                    << sseq_a_[i] << ", "
                    << sseq_b_[i];
                if (numel > 0)
                {
                    std::cout << "), (";
                }
                else
                {
                    break;
                }
            }
            std::cout << ")" << std::endl << std::flush;
        }

    private:
        std::size_t sseqs_size_{};
        T distance_ = std::numeric_limits<T>::infinity();
        std::size_t start_a_{};
        std::size_t start_b_{};
        I size_a_{};
        I size_b_{};
        std::vector<T> sseq_a_{};
        std::vector<T> sseq_b_{};
        std::map<S, std::size_t, std::greater<S>> pairs_distances_{};
        S tol_ = S(0);
        std::vector<S> memo_distances_{};
        std::vector<I> memo_sizes_{};

        auto lcs_impl(T const *a, I sa, T const *b, I sb) -> std::pair<S, I>
        {
            if ((sa == 0) || (sb == 0))
            {
                return std::make_pair(S(0), static_cast<I>(0));
            }

            S dist = memo_dist(sa, sb);
            I size = memo_size(sa, sb);

            auto [daa, saa] = lcs_impl(a + 1, sa - 1, b, sb);
            auto [dab, sab] = lcs_impl(a, sa, b + 1, sb - 1);

        }

        bool equal(T lhs, T rhs)
        {
            if (abs(lhs - rhs) <= tol_)
            {
                return true;
            }

            return false; 
        }

        S memo_dist(I i, I j)
        {
            S x = S(0);
            if ((i >= 0) && (i < size_a_) && (j >= 0) && (j < size_b_))
            {
                x = memo_distances_[i + size_a_ * j];
            }

            return x;
        }

        I memo_size(I i, I j)
        {
            I x = I(0);
            if ((i >= 0) && (i < size_a_) && (j >= 0) && (j < size_b_))
            {
                x = memo_sizes_[i + size_a_ * j];
            }

            return x;
        }
};

template<typename T, typename I = int64_t>
class closest_largest_subsequences {
    public:
        using S = decltype(std::real(T{}));

        [[maybe_unused]] auto operator()(T *a, I size_a, T *b, I size_b, S tol) -> /**! Size of subsequences */ I
        {
            std::lock_guard<std::mutex> lock(m_);

            if ((size_a > 0) && (size_b > 0) && (tol >= 0))
            {
                // Initialize members
                clear();
                this->tol_= tol;
                this->size_a_ = size_a;
                this->size_b_ = size_b;
                memo_distances_.resize(size_a * size_b, S(-1));
                memo_sizes_.resize(size_a * size_b, I(-1));
                memo_next_.resize(size_a * size_b, I(-1));

                // Call recursive, memoized implementation to compute subsequences
                auto [distance, sseqs_size, _] = clss_implr(a, size_a - 1, b, size_b - 1);
                this->distance_ = distance;
                this->sseqs_size_ = sseqs_size;

                // Extract `sseq_a_` and `sseq_b_` from `a` and `b` and set:
                // --> inf_norm_ = ||sseq_a_ - sseq_b_||_inf
                extract_subsequences(a, size_a, b, size_b);
            }

            return sseqs_size_;
        }

        auto distance() -> S
        {
            std::lock_guard<std::mutex> lock(m_);
            return distance_;
        }

        auto inf_norm() -> S
        {
            std::lock_guard<std::mutex> lock(m_);
            return inf_norm_;
        }

        auto subseqs_ids() -> std::pair<std::vector<S>, std::vector<S>>
        {
            std::lock_guard<std::mutex> lock(m_);
            return std::make_pair(sseq_a_ids_, sseq_b_ids_);
        }

        auto subseqs() -> std::pair<std::vector<S>, std::vector<S>>
        {
            std::lock_guard<std::mutex> lock(m_);
            return std::make_pair(sseq_a_, sseq_b_);
        }

        auto subseqs_size() -> I
        {
            std::lock_guard<std::mutex> lock(m_);
            return sseqs_size_;
        }

        ///!
        ///! The methods that follow are meant for use while debugging
        ///!

        void print_pairs(I num_pairs = 0)
        {
            std::lock_guard<std::mutex> lock(m_);

            if (pairs_distances_.empty())
            {
                for (I i = 0; i < sseqs_size_; ++i)
                {
                    T d_ = std::abs(sseq_a_[i] - sseq_b_[i]);
                    pairs_distances_.insert({d_, i});
                }
            }

            std::cout << ":: :: Accumulated distance: " << distance_;
            std::cout << ", subsequences pairs (largest to smallest distance): (";
            num_pairs = (num_pairs == 0) ? pairs_distances_.size() : num_pairs;
            int numel = static_cast<int>(std::min(num_pairs, I(pairs_distances_.size())));
            for (auto& [_, i] : pairs_distances_)
            {
                numel--;
                std::cout << ""
                    << sseq_a_[i] << ", "
                    << sseq_b_[i];
                if (numel > 0)
                {
                    std::cout << "), (";
                }
                else
                {
                    break;
                }
            }
            std::cout << ")" << std::endl << std::flush;
        }

        void print_extract_subsequences()
        {
            std::lock_guard<std::mutex> lock(m_);

            std::cout << ">>> Traversing:";
            I sa = size_a_ - 1;
            I sb = size_b_ - 1;
            I index = ij2index(sa, sb);
            if (!in_range(index) || (sseqs_size_ == 0))
            {
                std::cout << " nothing to print\n";
                return;
            }
            std::cout << std::endl;

            I next_index = index, i = 0;
            do
            {
                index = next_index;
                next_index = memo_next(index);
                next_index = in_range(next_index) ? next_index : index;

                I ia, ib;
                I si = memo_size(index);
                I nsi = memo_size(next_index);
                if ((nsi < si) || (index == next_index))
                {
                    auto [ja, jb] = index2ij(index);

                    ia = sa - ja;
                    ib = sb - jb;

                    std::cout << ""
                        << ":: :: Indices: (" << ia << ", " << ib 
                        << ") :: Elements: (" << sseq_a_[i] << ", " << sseq_b_[i] 
                        << ") :: (acc dist = " << memo_dist(ja, jb) << ", size = " << memo_size(ja, jb)
                        << ")\n";
                    ++i;
                }
            } while ((index != next_index) && in_range(index));
        }

        void debug(T *a, T *b, I num = 0)
        {
            auto print_input_sequences = [](auto& a, auto a_size, auto& b, auto b_size) {
                std::cout << ">>> Input: \n";

                std::cout << ":: :: a = {";
                for (std::size_t i = 0; i < a_size; ++i)
                {
                    std::cout << a[i];
                    if (i != a_size - 1)
                    {
                        std::cout << ", ";
                    }
                    else
                    {
                    }

                }
                std::cout << "}\n\n";

                std::cout << ":: :: b = {";
                for (std::size_t i = 0; i < b_size; ++i)
                {
                    std::cout << b[i];
                    if (i != b_size - 1)
                    {
                        std::cout << ", ";
                    }
                    else
                    {
                    }

                }
                std::cout << "}\n\n";
            };

            print_input_sequences(a, size_a_, b, size_b_);
            std::cout << ":: :: tol = " << tol_ << std::endl << std::endl;
            closest_largest_subsequences<float> lcs;
            std::cout << ">>>>>>>>>>>>\n";
            std::cout << ":: :: Subsequences sub_a, sub_b have distance: " << distance_ << ", size: " << sseqs_size_ 
                << ", and ||sub_a - sub_b||_inf = " << inf_norm_ << std::endl << std::endl;

            /* print_pairs(num); */
            /* std::cout << std::endl; */
            print_extract_subsequences();
            std::cout << "<<<<<<<<<<<<\n";
        };


    private:
        S tol_ = S(0);
        I sseqs_size_{};
        S distance_ = std::numeric_limits<S>::infinity();
        S inf_norm_ = std::numeric_limits<S>::infinity();
        I size_a_{};
        I size_b_{};
        std::vector<T> sseq_a_{};
        std::vector<T> sseq_b_{};
        std::vector<T> sseq_a_ids_{};
        std::vector<T> sseq_b_ids_{};
        std::map<S, I, std::greater<S>> pairs_distances_{};
        std::vector<S> memo_distances_{};
        std::vector<I> memo_sizes_{};
        std::vector<I> memo_next_{};
        std::mutex m_;

        void clear()
        {
            tol_ = S(0);
            sseqs_size_ = {};
            distance_ = std::numeric_limits<T>::infinity();
            S inf_norm_ = std::numeric_limits<S>::infinity();
            size_a_ = {};
            size_b_ = {};
            sseq_a_ = {};
            sseq_b_ = {};
            sseq_a_ids_ = {};
            sseq_b_ids_ = {};
            pairs_distances_ = {};
            memo_distances_ = {};
            memo_sizes_ = {};
            memo_next_ = {};
        }

        ///! Recursive implementation with memoization
        auto clss_implr(T const *a, I sa, T const *b, I sb) -> std::tuple</* acc distance */ S, /* size */ I, /* next */ I>
        {
            //
            // Base case: at least one of the sequences is empty
            //
            if (!in_range(sa, sb))
            {
                return std::make_tuple(std::numeric_limits<S>::infinity(), I(0), I(-1));
            }

            //
            // If `dist`, `size` and `next_index` have already been computed for this pair of `sa`, `sb` return
            //
            auto [dist, size, next_index] = memo(sa, sb);
            if (memo_valid(dist, size))
            {
                return std::make_tuple(dist, size, next_index);
            }

            //
            // Otherwise, compute new `dist`, `size` and `next_index`
            //

            // Initialize local vars
            dist = std::numeric_limits<S>::infinity();
            size = I(0);
            auto do_update = [](S d, I s, I nindex, S& dist, I& size, I& next_index) -> bool {
                bool update = false;
                if (size < s)
                {
                    dist = d;
                    size = s;
                    next_index = nindex;
                    update = true;
                }
                else if(size == s)
                {
                    if (dist > d)
                    {
                        dist = d;
                        next_index = nindex;
                        update = true;
                    }
                }

                return update;
            };
            [[maybe_unused]] bool update = false;

            // Case 1: a[0] .==. b[0], try to match next element of sequence `a` with next element of sequence `b`
            if(equal(a[0], b[0]))
            {
                auto [d, s, nindex] = clss_implr(a + 1, sa - 1, b + 1, sb - 1);
                if (d == std::numeric_limits<S>::infinity())
                {
                    dist = std::abs(a[0] - b[0]);
                    size = 1;
                    next_index = ij2index(sa, sb);
                    update = true;
                }
                else
                {
                    d += std::abs(a[0] - b[0]);
                    ++s;
                    update = do_update(d, s, nindex, dist, size, next_index);
                }

            }

            // Case 2: a[0] not equivalent to b[0], try to match next element of sequence `a` with current element of sequence `b`
            {
                auto [d, s, nindex] = clss_implr(a + 1, sa - 1, b, sb);
                do_update(d, s, nindex, dist, size, next_index);
            }

            // Case 3: a[0] not equivalent to b[0], try to match current element of sequence `a` with next element of sequence `b`
            {
                auto [d, s, nindex] = clss_implr(a, sa, b + 1, sb - 1);
                do_update(d, s, nindex, dist, size, next_index);
            }

            // Save best results from 3 cases
            memo_dist(sa, sb) = dist;
            memo_size(sa, sb) = size;
            memo_next(sa, sb) = next_index;

            // Make next pair point to this one
            if (update)
            {
                next_index = ij2index(sa, sb);
            }

            return std::make_tuple(dist, size, next_index);
        }

        void extract_subsequences(T* a, I size_a, T* b, I size_b)
        {
            I sa = size_a - 1;
            I sb = size_b - 1;

            I index = ij2index(sa, sb);
            if (!in_range(index) || (sseqs_size_ == 0))
            {
                return;
            }

            I next_index = index;
            inf_norm_ = S(0);
            do
            {
                index = next_index;
                next_index = memo_next(index);
                next_index = in_range(next_index) ? next_index : index;

                I ia, ib;
                I si = memo_size(index);
                I nsi = memo_size(next_index);
                if ((nsi < si) || (index == next_index))
                {
                    auto [ja, jb] = index2ij(index);

                    ia = sa - ja;
                    sseq_a_ids_.push_back(ia);
                    sseq_a_.push_back(a[ia]);

                    ib = sb - jb;
                    sseq_b_ids_.push_back(ib);
                    sseq_b_.push_back(b[ib]);

                    S norm = std::abs(a[ia] - b[ib]);
                    inf_norm_ = std::max(inf_norm_, norm);
                }
            } while ((index != next_index) && in_range(index));
            sseqs_size_ = sseq_a_.size();
        }

        ///!
        ///! Helper functions
        ///!
        bool equal(T lhs, T rhs) const
        {
            if (std::abs(lhs - rhs) <= tol_)
            {
                return true;
            }

            return false; 
        }

        bool in_range(I i, I j) const
        {
            bool in_range = false;

            if ((i >= 0) && (i < size_a_) && (j >= 0) && (j < size_b_))
            {
                in_range = true;
            }

            return in_range;
        }

        bool in_range(I index) const
        {
            bool in_range = false;

            I upper_bound = size_a_ * size_b_;
            if ((index >= 0) && (index < upper_bound))
            {
                in_range = true;
            }

            return in_range;
        }

        auto memo(I i, I j) -> std::tuple<S, I, I> const
        {
            auto d = memo_dist(i, j);
            auto s = memo_size(i, j);
            auto n = memo_next(i, j);

            return std::make_tuple(d, s, n);
        }

        S memo_dist(I i, I j) const &&
        {
            auto x = memo_distances_[ij2index(i, j)];
            return x;
        }

        S& memo_dist(I i, I j) &
        {
            auto& x = memo_distances_[ij2index(i, j)];
            return x;
        }

        I memo_size(I i, I j) const &&
        {
            auto x = memo_sizes_[ij2index(i, j)];
            return x;
        }

        I& memo_size(I i, I j) &
        {
            auto& x = memo_sizes_[ij2index(i, j)];
            return x;
        }

        I memo_size(I index) const &&
        {
            auto x = memo_sizes_[index];
            return x;
        }

        I& memo_size(I index) &
        {
            auto& x = memo_sizes_[index];
            return x;
        }

        I memo_next(I i, I j) const &&
        {
            auto x = memo_next_[ij2index(i, j)];
            return x;
        }        

        I& memo_next(I i, I j) &
        {
            auto& x = memo_next_[ij2index(i, j)];
            return x;
        }

        I memo_next(I index) const &&
        {
            auto x = memo_next_[index];
            return x;
        } 

        I& memo_next(I index) &
        {
            auto& x = memo_next_[index];
            return x;
        } 

        bool memo_valid(S d, I s) const
        {
            bool valid = true;
            if ((d == -1) || (s == -1))
            {
                valid = false;
            }

            return valid;
        }

        auto ij2index(I i, I j) -> I const
        {
            return i + size_a_ * j;
        }

        auto index2ij(I index) -> std::pair<I, I> const
        {
            I i = index % size_a_;
            I j = (index - i) / size_a_;
            return std::make_pair(i, j);
        }
};

#include "common/misc/lapack_host_reference.hpp"

// Compute eigenvalues and eigenvectors of A with lapack_*syev
template<typename T, typename S>
bool eig(const rocblas_fill uplo, T const *A, const int n, T *U, S *D)
{
    if (A == nullptr || n < 1)
    {
        return false;
    }
    [[maybe_unused]] volatile auto mptr = memcpy(U, A, n * n * sizeof(T));

    int info;
    int worksize = n * n;
    std::vector<T> work(worksize, T(0.));
    int worksize_real = n * n;
    std::vector<S> work_real(worksize_real, S(0.));
    cpu_syev_heev(rocblas_evect_original, uplo, n, U, n, D, work.data(), worksize, work_real.data(), worksize_real, &info);

    return (info == 0);
}

template<typename T, typename S>
bool eig(const rocblas_fill uplo, T const *A, const int n, std::vector<T> &U, std::vector<S> &D)
{
    if (A == nullptr || n < 1)
    {
        return false;
    }

    D.resize(n, S(0.));
    U.resize(n * n, T(0.));

    return eig(uplo, A, n, D.data(), U.data());
}

// Form matrix Y = X * D * X^*, where X^* is the adjoint of X
// X is nrowsX x dimD; D is dimD x dimD; Y is nrowsX x nrowsX
template<typename T, typename S>
bool XDXh(T const *X, const int nrowsX, S const *D, const int dimD, T *Y)
{
    if (X == nullptr || D == nullptr)
    {
        return false;
    }

    const int ldX = nrowsX;
    constexpr bool T_is_complex = rocblas_is_complex<T>;
    auto rocblas_operation_adjoint = T_is_complex ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;

    std::vector<T> W(dimD * dimD, T(0.)), Z(nrowsX * dimD, T(0.));
    const int ldZ = nrowsX;
    const int ldU = dimD;
    for (int i = 0; i < dimD; ++i)
    {
        W[i + i * dimD] = D[i];
    }

    cpu_gemm(rocblas_operation_none, rocblas_operation_none, nrowsX, dimD, dimD, T(1.), const_cast<T*>(X), nrowsX, W.data(), dimD, T(0.), Z.data(), dimD);
    cpu_gemm(rocblas_operation_none, rocblas_operation_adjoint, nrowsX, dimD, nrowsX, T(1.), Z.data(), nrowsX, const_cast<T*>(X), dimD, T(0.), Y, dimD);
    return true;
}

template<typename T, typename S>
bool XDXh(T const *X, const int nrows, S const *D, const int dimD, std::vector<T> &Y)
{
    Y.resize(dimD * dimD, T(0.));
    return XDXh(X, nrows, D, dimD, Y.data());
}

// Form matrix Y = X^* * X - I, where X^* is the adjoint of X
// X is nrows x ncols; Y is nrows x nrows
template<typename T>
bool XhXminusI(T const *X, const int nrows, const int ncols, std::vector<T> &Y)
{
    if (X == nullptr)
    {
        return false;
    }

    auto rocblas_operation_adjoint = rocblas_is_complex<T> ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;
    Y.resize(ncols * ncols, T(0.));
    cpu_gemm(rocblas_operation_adjoint, rocblas_operation_none, ncols, nrows, ncols, T(1.), X, ncols, X, nrows, T(0.), Y.data(), ncols);

    for (int i = 0; i < ncols; ++i)
    {
        Y[i + i * ncols] -= T(1.);
    }

    return true;
}

//
// Given inputs X (size nrowsX x dimD) and D (size dimD):
//
// Form matrix Y = U * diag(D) * U^*, where
// - U is the unitary matrix obtained from the QR decomposition of X,
// - U^* is the adjoint of U.
//
// Output Y is nrowsU x nrowsU
//
template<typename T, typename S>
bool UDUh(T const *X, const int nrowsX, S const *D, const int dimD, T *Y)
{
    const int ncolsX = dimD;
    const int ldX = nrowsX;
    const int m = std::max(nrowsX, ncolsX);

    int info;
    int worksize = 1;
    std::vector<T> work(worksize, T(0.)); // lapack workspace
    std::vector<T> tau(m); // scalar factors of geqrf reflectors
    auto rocblas_operation_adjoint = rocblas_is_complex<T> ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;

    //
    // Create diagonal matrix W = diag(D)
    //
    const int dimW = dimD;
    std::vector<T> W(dimW * dimW, T(0.)), Z(dimW * dimW, T(0.));
    for (int i = 0; i < dimW; ++i)
    {
        W[i + i * dimW] = D[i];
    }

    //
    // Extract unitary matrix
    //
    std::vector<T> U(nrowsX * ncolsX, T(0.));
    { [[maybe_unused]] auto mptr = memcpy(U.data(), X, nrowsX * ncolsX * sizeof(T)); }

    // Pick something that is big enough for work size of geqrf
    worksize = m * m;
    work.resize(worksize, T(0.));
    const int nrowsU = nrowsX;
    const int ncolsU = ncolsX;
    const int ldU = ldX;
    cpu_geqrf<T>(nrowsU, ncolsU, U.data(), ldX, tau.data(), work.data(), worksize);

    /* // Infer work size of [or,un]mqr */
    /* worksize = -1; */
    /* cpu_ormqr_unmqr<T>(rocblas_side_right, rocblas_operation_adjoint, nrowsU, ncolsU, dimW, U.data(), ldU, tau.data(), W.data(), dimW, work.data(), worksize, &info); */
    info = -1;

    if (info == 0)
    {
        // Use LAPACK's suggested work size
        worksize = std::real(work[0]);
    }
    else
    {
        // Pick something that is big enough for work size
        worksize = m * m;
    }
    work.resize(worksize, T(0.));

    //
    // Create matrix: Y = U * diag(D) * U^*
    //
    cpu_ormqr_unmqr<T>(rocblas_side_left, rocblas_operation_none, nrowsU, ncolsU, dimW, U.data(), ldU, tau.data(), W.data(), dimW, work.data(), worksize, &info);
    cpu_ormqr_unmqr<T>(rocblas_side_right, rocblas_operation_adjoint, nrowsU, ncolsU, dimW, U.data(), ldU, tau.data(), W.data(), dimW, work.data(), worksize, &info);
    { [[maybe_unused]] auto mptr = memcpy(Y, W.data(), W.size() * sizeof(T)); }

    return true;
}
