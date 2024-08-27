#pragma once

#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <vector>

template <typename T,
          typename I = std::int64_t,
          typename = typename std::enable_if<std::is_signed<std::decay_t<I>>::value>::type>
class closest_largest_subsequences
{
public:
    using S = decltype(std::real(T{}));

    [[maybe_unused]] auto operator()(T const* a, I size_a, T const* b, I size_b, S tol)
        -> /**! Size of subsequences */ I
    {
        std::lock_guard<std::mutex> lock(m_);

        clear();
        if((size_a > 0) && (size_b > 0) && (tol >= 0))
        {
            //
            // Initialize members
            //
            this->tol_ = tol;
            this->size_a_ = size_a;
            this->size_b_ = size_b;
            this->memo_distances_.resize(size_a * size_b, std::numeric_limits<S>::infinity());
            this->memo_sizes_.resize(size_a * size_b, S(-1));
            this->memo_next_.resize(size_a * size_b, I(-1));
            // Copy original sequences for debugging purposes
            this->seq_a_.resize(size_a, T(0));
            [[maybe_unused]] auto volatile a_mptr = memcpy(seq_a_.data(), a, sizeof(T) * size_a);
            this->seq_b_.resize(size_b, T(0));
            [[maybe_unused]] auto volatile b_mptr = memcpy(seq_b_.data(), b, sizeof(T) * size_b);

            //
            // Call recursive, memoized, implementation to compute subsequences
            //
            auto [distance, sseqs_size, _] = clss_implr(a, size_a - 1, b, size_b - 1);
            this->distance_ = distance;
            this->sseqs_size_ = sseqs_size;

            //
            // Extract `sseq_a_` and `sseq_b_` from `a` and `b` and set:
            // inf_norm_ = ||sseq_a_ - sseq_b_||_inf
            //
            this->inf_norm_ = extract_subsequences(a, size_a, b, size_b);
        }

        return sseqs_size_;
    }

    [[maybe_unused]] auto operator()(const std::vector<T>& a, const std::vector<T>& b, S tol)
        -> /**! Size of subsequences */ I
    {
        return this->operator()(a.data(), static_cast<I>(a.size()), b.data(),
                                static_cast<I>(b.size()), tol);
    }

    [[maybe_unused]] auto operator()(T* a, I size_a, T* b, I size_b, S tol)
        -> /**! Size of subsequences */ I
    {
        return this->operator()(const_cast<T const*>(a), size_a, const_cast<T const*>(b), size_b,
                                tol);
    }

    template <typename J, typename = typename std::enable_if<std::is_integral<std::decay_t<J>>::value>::type>
    [[maybe_unused]] auto operator()(T const* a, J size_a, T const* b, J size_b, S tol)
        -> /**! Size of subsequences */ I
    {
        return this->operator()(a, static_cast<I>(size_a), b, static_cast<I>(size_b), tol);
    }

    template <typename J, typename = typename std::enable_if<std::is_integral<std::decay_t<J>>::value>::type>
    [[maybe_unused]] auto operator()(T* a, J size_a, T* b, J size_b, S tol)
        -> /**! Size of subsequences */ I
    {
        return this->operator()(const_cast<T const*>(a), static_cast<I>(size_a),
                                const_cast<T const*>(b), static_cast<I>(size_b), tol);
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

    ///
    /// The methods that follow are meant for use while debugging
    ///

    void print_debug()
    {
        std::lock_guard<std::mutex> lock(m_);

        auto a = seq_a_.data();
        auto b = seq_b_.data();

        const auto default_precision{std::cout.precision()};
        const auto digits
            = static_cast<I>(tol_ > S(0) ? std::ceil(-std::min(std::log10(tol_), S(0))) + 2
                                         : std::numeric_limits<T>::max_digits10);
        std::cout << std::fixed << std::setprecision(digits);

        auto print_input_sequences = [](auto& a, auto a_size, auto& b, auto b_size) {
            std::cout << ">>> Input: \n";

            std::cout << ":: :: a = {";
            for(I i = 0; i < a_size; ++i)
            {
                std::cout << a[i];
                if(i != a_size - 1)
                {
                    std::cout << ", ";
                }
                else
                {
                }
            }
            std::cout << "}\n\n";

            std::cout << ":: :: b = {";
            for(I i = 0; i < b_size; ++i)
            {
                std::cout << b[i];
                if(i != b_size - 1)
                {
                    std::cout << ", ";
                }
                else
                {
                }
            }
            std::cout << "}\n\n";
        };

        std::cout << ">>>>>>>>>>>>\n";
        std::cout << ":: :: closest_largest_subsequences::print_debug()\n\n" << std::flush;
        print_input_sequences(a, size_a_, b, size_b_);
        std::cout << ":: :: tol = " << tol_ << std::endl << std::endl;

        std::cout << "++++++++++++\n";
        std::cout << ":: :: Subsequences sub_a, sub_b have distance: " << distance_
                  << ", size: " << sseqs_size_ << ", and ||sub_a - sub_b||_inf = " << inf_norm_
                  << std::endl
                  << std::endl
                  << std::flush;

        print_extract_subsequences();
        std::cout << "<<<<<<<<<<<<\n" << std::flush;

        // Restore defaults
        std::cout << std::setprecision(default_precision);
    }

private:
    S tol_{};
    I sseqs_size_{};
    S distance_ = std::numeric_limits<S>::infinity();
    S inf_norm_ = std::numeric_limits<S>::infinity();
    I size_a_{};
    I size_b_{};
    std::vector<T> seq_a_{};
    std::vector<T> seq_b_{};
    std::vector<T> sseq_a_{};
    std::vector<T> sseq_b_{};
    std::vector<T> sseq_a_ids_{};
    std::vector<T> sseq_b_ids_{};
    std::vector<S> memo_distances_{};
    std::vector<I> memo_sizes_{};
    std::vector<I> memo_next_{};
    std::mutex m_;

    void clear()
    {
        tol_ = {};
        sseqs_size_ = {};
        distance_ = std::numeric_limits<T>::infinity();
        inf_norm_ = std::numeric_limits<S>::infinity();
        size_a_ = {};
        size_b_ = {};
        seq_a_ = {};
        seq_b_ = {};
        sseq_a_ = {};
        sseq_b_ = {};
        sseq_a_ids_ = {};
        sseq_b_ids_ = {};
        memo_distances_ = {};
        memo_sizes_ = {};
        memo_next_ = {};
    }

    /// Recursive implementation with memoization
    auto clss_implr(T const* a, I sa, T const* b, I sb)
        -> std::tuple</* acc distance */ S, /* size */ I, /* next */ I>
    {
        //
        // Base case: at least one of the sequences is empty
        //
        if(!in_range(sa, sb))
        {
            return std::make_tuple(std::numeric_limits<S>::infinity(), I(0), I(-1));
        }

        //
        // If `dist`, `size` and `next_index` have already been computed for this pair of `sa`, `sb` return
        //
        auto [dist, size, _] = memo(sa, sb);
        I next_index = I(-1);

        if(memo_valid(dist, size))
        {
            // Make next entry point to this one
            next_index = ij2index(sa, sb);

            return std::make_tuple(dist, size, next_index);
        }

        //
        // Otherwise, compute new `dist`, `size` and `next_index`
        //

        // Initialize local vars
        dist = std::numeric_limits<S>::infinity();
        size = I(0);
        // Compare current optimum (dist, size) with candidate optimum (d, s), and update if necessary
        auto do_update = [](S d, I s, I nindex, S& dist, I& size, I& next_index) -> bool {
            bool update = false;
            if(size < s)
            {
                dist = d;
                size = s;
                next_index = nindex;
                update = true;
            }
            else if(size == s)
            {
                if(dist > d)
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
        if(equiv(a[0], b[0]))
        {
            auto [d, s, nindex] = clss_implr(a + I(1), sa - I(1), b + I(1), sb - I(1));
            if(d == std::numeric_limits<S>::infinity())
            {
                dist = std::abs(a[0] - b[0]);
                size = I(1);
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

        // Case 2: try to match next element of sequence `a` with current element of sequence `b`
        {
            auto [d, s, nindex] = clss_implr(a + I(1), sa - I(1), b, sb);
            update = do_update(d, s, nindex, dist, size, next_index);
        }

        // Case 3: try to match current element of sequence `a` with next element of sequence `b`
        {
            auto [d, s, nindex] = clss_implr(a, sa, b + I(1), sb - I(1));
            update = do_update(d, s, nindex, dist, size, next_index);
        }

        // Save best results from 3 cases
        memo_dist(sa, sb) = dist;
        memo_size(sa, sb) = size;
        memo_next(sa, sb) = next_index;

        // Make next entry point to this one
        next_index = ij2index(sa, sb);

        return std::make_tuple(dist, size, next_index);
    }

    auto extract_subsequences(T const* a, I size_a, T const* b, I size_b)
        -> /* || sseq_a_ - sseq_b_ ||_inf */ S
    {
        S inf_norm = std::numeric_limits<S>::infinity();
        I sa = size_a - I(1);
        I sb = size_b - I(1);

        I index = ij2index(sa, sb);
        if(!in_range(index) || (sseqs_size_ == I(0)))
        {
            return inf_norm;
        }

        I next_index = index;
        inf_norm = static_cast<S>(0);
        do
        {
            index = next_index;
            next_index = memo_next(index);
            next_index = in_range(next_index) ? next_index : index;

            I ia, ib;
            I si = memo_size(index);
            I nsi = memo_size(next_index);
            if((nsi < si) || (index == next_index))
            {
                auto [ja, jb] = index2ij(index);

                ia = sa - ja;
                sseq_a_ids_.push_back(ia);
                sseq_a_.push_back(a[ia]);

                ib = sb - jb;
                sseq_b_ids_.push_back(ib);
                sseq_b_.push_back(b[ib]);

                S norm = std::abs(a[ia] - b[ib]);
                inf_norm = std::max(inf_norm, norm);
            }
        } while((index != next_index) && in_range(index));

        return inf_norm;
    }

    void print_extract_subsequences()
    {
        std::cout << ">>> Traversing:";
        I sa = size_a_ - I(1);
        I sb = size_b_ - I(1);
        I index = ij2index(sa, sb);
        if(!in_range(index) || (sseqs_size_ == I(0)))
        {
            std::cout << " nothing to print\n";
            return;
        }
        std::cout << std::endl;

        I next_index = index, i = I(0);
        do
        {
            index = next_index;
            next_index = memo_next(index);
            next_index = in_range(next_index) ? next_index : index;

            I ia, ib;
            I si = memo_size(index);
            I nsi = memo_size(next_index);
            if((nsi < si) || (index == next_index))
            {
                auto [ja, jb] = index2ij(index);

                ia = sa - ja;
                ib = sb - jb;

                std::cout << ""
                          << ":: :: Indices: (" << ia << ", " << ib << ") :: Elements: ("
                          << sseq_a_[i] << ", " << sseq_b_[i]
                          << ") :: (acc dist = " << memo_dist(ja, jb)
                          << ", size = " << memo_size(ja, jb) << ")\n";
                ++i;
            }
        } while((index != next_index) && in_range(index));
    }

    ///
    /// Helper functions
    ///

    /// lhs, rhs are "equivalent" (symbolyc notation: lhs .=. rhs)
    /// when |lhs - rhs| <= tol.
    ///
    /// This is not a true equivalence relation.
    bool equiv(T lhs, T rhs) const
    {
        if(std::abs(lhs - rhs) <= tol_)
        {
            return true;
        }

        return false;
    }

    bool in_range(I i, I j) const
    {
        bool in_range = false;

        if((i >= 0) && (i < size_a_) && (j >= I(0)) && (j < size_b_))
        {
            in_range = true;
        }

        return in_range;
    }

    bool in_range(I index) const
    {
        bool in_range = false;

        I upper_bound = size_a_ * size_b_;
        if((index >= I(0)) && (index < upper_bound))
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

    S memo_dist(I i, I j) const&&
    {
        auto x = memo_distances_[ij2index(i, j)];
        return x;
    }

    S& memo_dist(I i, I j) &
    {
        auto& x = memo_distances_[ij2index(i, j)];
        return x;
    }

    I memo_size(I i, I j) const&&
    {
        auto x = memo_sizes_[ij2index(i, j)];
        return x;
    }

    I& memo_size(I i, I j) &
    {
        auto& x = memo_sizes_[ij2index(i, j)];
        return x;
    }

    I memo_size(I index) const&&
    {
        auto x = memo_sizes_[index];
        return x;
    }

    I& memo_size(I index) &
    {
        auto& x = memo_sizes_[index];
        return x;
    }

    I memo_next(I i, I j) const&&
    {
        auto x = memo_next_[ij2index(i, j)];
        return x;
    }

    I& memo_next(I i, I j) &
    {
        auto& x = memo_next_[ij2index(i, j)];
        return x;
    }

    I memo_next(I index) const&&
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
        if((d == S(-1)) || (s == I(-1)))
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
