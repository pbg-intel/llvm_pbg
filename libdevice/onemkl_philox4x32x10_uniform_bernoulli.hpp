/*******************************************************************************
* Copyright (C) 2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#ifndef _MKL_RNG_DEVICE_PHILOX4X32X10_UNIFORM_BERNOULLI_IMPL_HPP_
#define _MKL_RNG_DEVICE_PHILOX4X32X10_UNIFORM_BERNOULLI_IMPL_HPP_

#include <stdexcept>
#include <algorithm>
#include <utility>
#include <limits>
#include <cmath>

namespace oneapi::mkl::rng::device {

class philox4x32x10;

namespace detail {

// internal structure to specify state of engine
template <typename EngineType>
struct engine_state {};

template <typename EngineType>
class engine_base {};

template <typename DistrType>
class distribution_base {};

} // namespace detail

// METHODS FOR DISTRIBUTIONS

namespace uniform_method {
struct standard {};
struct accurate {};
using by_default = standard;
} // namespace uniform_method

namespace bernoulli_method {
struct icdf {};
using by_default = icdf;
} // namespace bernoulli_method

// GENERATE FUNCTIONS

template <typename Distr, typename Engine>
typename Distr::result_type generate(Distr& distr, Engine& engine) {
    return distr.generate(engine);
}

template <typename Distr, typename Engine>
typename Distr::result_type generate_single(Distr& distr, Engine& engine) {
    return distr.generate_single(engine);
}

// SERVICE FUNCTIONS

template <typename Engine>
void skip_ahead(Engine& engine, std::uint64_t num_to_skip) {
    engine.skip_ahead(num_to_skip);
}

template <typename Engine>
void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip) {
    engine.skip_ahead(num_to_skip);
}

// CONTINUOUS AND DISCRETE RANDOM NUMBER DISTRIBUTIONS

template <typename Type, typename Method = uniform_method::by_default>
class uniform : detail::distribution_base<uniform<Type, Method>> {
public:
    static_assert(std::is_same<Method, uniform_method::standard>::value ||
                      std::is_same<Method, uniform_method::accurate>::value,
                  "oneMKL: rng/uniform: method is incorrect");

    static_assert(std::is_same<Type, std::int32_t>::value,
                  "oneMKL: rng/uniform: only int32_t is supported");

    using method_type = Method;
    using result_type = Type;
    using param_type = typename detail::distribution_base<uniform<Type, Method>>::param_type;

    uniform() : detail::distribution_base<uniform<Type, Method>>(Type(0.0),
        std::is_same<Method, uniform_method::standard>::value ? (1 << 23) : std::numeric_limits<Type>::max()) {}

    explicit uniform(Type a, Type b) : detail::distribution_base<uniform<Type, Method>>(a, b) {}
    explicit uniform(const param_type& pt)
            : detail::distribution_base<uniform<Type, Method>>(pt.a_, pt.b_) {}

    Type a() const {
        return detail::distribution_base<uniform<Type, Method>>::a();
    }

    Type b() const {
        return detail::distribution_base<uniform<Type, Method>>::b();
    }

    param_type param() const {
        return detail::distribution_base<uniform<Type, Method>>::param();
    }

    void param(const param_type& pt) {
        detail::distribution_base<uniform<Type, Method>>::param(pt);
    }

private:
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate(Distr& distr, Engine& engine);

    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

template <typename IntType, typename Method = bernoulli_method::by_default>
class bernoulli : detail::distribution_base<bernoulli<IntType, Method>> {
public:
    static_assert(std::is_same<Method, bernoulli_method::icdf>::value,
                  "oneMKL: rng/bernoulli: method is incorrect");

    static_assert(std::is_same<IntType, std::int32_t>::value,
                  "oneMKL: rng/bernoulli: type is not supported");

    using method_type = Method;
    using result_type = IntType;
    using param_type = typename detail::distribution_base<bernoulli<IntType, Method>>::param_type;

    bernoulli() : detail::distribution_base<bernoulli<IntType, Method>>(0.5f) {}

    explicit bernoulli(float p) : detail::distribution_base<bernoulli<IntType, Method>>(p) {}
    explicit bernoulli(const param_type& pt)
            : detail::distribution_base<bernoulli<IntType, Method>>(pt.p_) {}

    float p() const {
        return detail::distribution_base<bernoulli<IntType, Method>>::p();
    }

    param_type param() const {
        return detail::distribution_base<bernoulli<IntType, Method>>::param();
    }

    void param(const param_type& pt) {
        detail::distribution_base<bernoulli<IntType, Method>>::param(pt);
    }

private:
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate(Distr& distr, Engine& engine);
    
    template <typename Distr, typename Engine>
    friend typename Distr::result_type generate_single(Distr& distr, Engine& engine);
};

// oneMKL Device API philox4x32x10 Implementation

namespace detail {

template <>
struct engine_state<oneapi::mkl::rng::device::philox4x32x10> {
    std::uint32_t key[2];
    std::uint32_t counter[4];
    std::uint32_t part;
    std::uint32_t result[4];
};

namespace philox4x32x10_impl {

static inline void add128(std::uint32_t* a, std::uint64_t b) {
    std::uint64_t tmp = ((static_cast<std::uint64_t>(a[1]) << 32) | a[0]);

    tmp += b;

    a[0] = static_cast<std::uint32_t>(tmp);
    a[1] = static_cast<std::uint32_t>(tmp >> 32);

    if (tmp < b) {
        tmp = ((static_cast<std::uint64_t>(a[3]) << 32) | a[2]) + 1;

        a[2] = static_cast<std::uint32_t>(tmp);
        a[3] = static_cast<std::uint32_t>(tmp >> 32);
    }
    return;
}

static inline void add128_1(std::uint32_t* a) {
    if (++a[0]) {
        return;
    }
    if (++a[1]) {
        return;
    }
    if (++a[2]) {
        return;
    }
    ++a[3];
}

static inline std::pair<std::uint32_t, std::uint32_t> mul_hilo_32(std::uint32_t a,
                                                                  std::uint32_t b) {
    std::uint64_t res_64 = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
    return std::make_pair(static_cast<std::uint32_t>(res_64),
                          static_cast<std::uint32_t>(res_64 >> 32));
}

static inline void round(std::uint32_t* cnt, std::uint32_t* k) {
    auto [L0, H0] = mul_hilo_32(0xD2511F53, cnt[0]);
    auto [L1, H1] = mul_hilo_32(0xCD9E8D57, cnt[2]);

    cnt[0] = H1 ^ cnt[1] ^ k[0];
    cnt[1] = L1;
    cnt[2] = H0 ^ cnt[3] ^ k[1];
    cnt[3] = L0;
}

static inline void round_10(std::uint32_t* cnt, std::uint32_t* k) {
    round(cnt, k); // 1
    // increasing keys with philox4x32x10 constants
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 2
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 3
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 4
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 5
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 6
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 7
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 8
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 9
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 10
}

static inline void skip_ahead(engine_state<oneapi::mkl::rng::device::philox4x32x10>& state,
                              std::uint64_t num_to_skip) {
    std::uint64_t num_to_skip_tmp = num_to_skip;
    std::uint64_t c_inc;
    std::uint32_t counter[4];
    std::uint32_t key[2];
    std::uint64_t tail;

    if (num_to_skip == 0) {
        return;
    }
    if (num_to_skip_tmp <= state.part) {
        state.part -= num_to_skip_tmp;
    }
    else {
        tail = num_to_skip % 4;
        if ((tail == 0) && (state.part == 0)) {
            add128(state.counter, num_to_skip / 4);
        }
        else {
            num_to_skip_tmp = num_to_skip_tmp - state.part;
            state.part = 0;
            c_inc = (num_to_skip_tmp - 1) / 4;
            state.part = (4 - num_to_skip_tmp % 4) % 4;
            add128(state.counter, c_inc);
            counter[0] = state.counter[0];
            counter[1] = state.counter[1];
            counter[2] = state.counter[2];
            counter[3] = state.counter[3];
            key[0] = state.key[0];
            key[1] = state.key[1];
            round_10(counter, key);
            state.result[0] = counter[0];
            state.result[1] = counter[1];
            state.result[2] = counter[2];
            state.result[3] = counter[3];
            add128_1(state.counter);
        }
    }
}

static inline void skip_ahead(engine_state<oneapi::mkl::rng::device::philox4x32x10>& state,
                              std::uint64_t n, const std::uint64_t* num_to_skip_ptr) {
    constexpr std::uint64_t uint_max = 0xFFFFFFFFFFFFFFFF;
    std::uint64_t post_buffer, pre_buffer;
    std::int32_t num_elements = 0;
    std::int32_t remained_counter;
    std::uint64_t tmp_skip_array[3] = { 0, 0, 0 };

    for (std::uint64_t i = 0; (i < 3) && (i < n); i++) {
        tmp_skip_array[i] = num_to_skip_ptr[i];
        if (tmp_skip_array[i]) {
            num_elements = i + 1;
        }
    }

    if (num_elements == 0) {
        return;
    }
    if ((num_elements == 1) && (tmp_skip_array[0] <= state.part)) {
        state.part -= static_cast<std::uint32_t>(tmp_skip_array[0]);
        return;
    }
    std::uint32_t counter[4];
    std::uint32_t key[2];

    if ((tmp_skip_array[0] - state.part) <= tmp_skip_array[0]) {
        tmp_skip_array[0] = tmp_skip_array[0] - state.part;
    }
    else if ((num_elements == 2) || (tmp_skip_array[1] - 1 < tmp_skip_array[1])) {
        tmp_skip_array[1] = tmp_skip_array[1] - 1;
        tmp_skip_array[0] = uint_max - state.part + tmp_skip_array[0];
    }
    else {
        tmp_skip_array[2] = tmp_skip_array[2] - 1;
        tmp_skip_array[1] = uint_max - 1;
        tmp_skip_array[0] = uint_max - state.part + tmp_skip_array[0];
    }

    state.part = 0;

    post_buffer = 0;

    remained_counter = static_cast<std::uint32_t>(tmp_skip_array[0] % 4);

    for (int i = num_elements - 1; i >= 0; i--) {
        pre_buffer = (tmp_skip_array[i] << 62);
        tmp_skip_array[i] >>= 2;
        tmp_skip_array[i] |= post_buffer;
        post_buffer = pre_buffer;
    }

    state.part = 4 - remained_counter;

    std::uint64_t counter64[] = { state.counter[1], state.counter[3] };
    counter64[0] = ((counter64[0] << 32ull) | state.counter[0]);
    counter64[1] = ((counter64[1] << 32ull) | state.counter[2]);

    counter64[0] += tmp_skip_array[0];

    if (counter64[0] < tmp_skip_array[0]) {
        counter64[1]++;
    }

    counter64[1] += tmp_skip_array[1];

    counter[0] = static_cast<std::uint32_t>(counter64[0]);
    counter[1] = static_cast<std::uint32_t>(counter64[0] >> 32);
    counter[2] = static_cast<std::uint32_t>(counter64[1]);
    counter[3] = static_cast<std::uint32_t>(counter64[1] >> 32);

    key[0] = state.key[0];
    key[1] = state.key[1];

    round_10(counter, key);

    state.result[0] = counter[0];
    state.result[1] = counter[1];
    state.result[2] = counter[2];
    state.result[3] = counter[3];

    counter64[0]++;

    if (counter64[0] < 1) {
        counter64[1]++;
    }

    state.counter[0] = static_cast<std::uint32_t>(counter64[0]);
    state.counter[1] = static_cast<std::uint32_t>(counter64[0] >> 32);
    state.counter[2] = static_cast<std::uint32_t>(counter64[1]);
    state.counter[3] = static_cast<std::uint32_t>(counter64[1] >> 32);
}

static inline void init(engine_state<oneapi::mkl::rng::device::philox4x32x10>& state,
                        std::uint64_t n, const std::uint64_t* seed_ptr, std::uint64_t offset) {
    state.key[0] = static_cast<std::uint32_t>(seed_ptr[0]);
    state.key[1] = static_cast<std::uint32_t>(seed_ptr[0] >> 32);

    state.counter[0] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1]) : 0);
    state.counter[1] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1] >> 32) : 0);

    state.counter[2] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2]) : 0);
    state.counter[3] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2] >> 32) : 0);

    state.part = 0;
    state.result[0] = 0;
    state.result[1] = 0;
    state.result[2] = 0;
    state.result[3] = 0;
    skip_ahead(state, offset);
}

static inline void init(engine_state<oneapi::mkl::rng::device::philox4x32x10>& state,
                        std::uint64_t n, const std::uint64_t* seed_ptr, std::uint64_t n_offset,
                        const std::uint64_t* offset_ptr) {
    state.key[0] = static_cast<std::uint32_t>(seed_ptr[0]);
    state.key[1] = static_cast<std::uint32_t>(seed_ptr[0] >> 32);

    state.counter[0] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1]) : 0);
    state.counter[1] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1] >> 32) : 0);

    state.counter[2] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2]) : 0);
    state.counter[3] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2] >> 32) : 0);

    state.part = 0;
    state.result[0] = 0;
    state.result[1] = 0;
    state.result[2] = 0;
    state.result[3] = 0;
    skip_ahead(state, n_offset, offset_ptr);
}

__attribute__((always_inline)) static inline std::uint32_t generate_single(
    engine_state<oneapi::mkl::rng::device::philox4x32x10>& state) {
    std::uint32_t res;

    std::uint32_t counter[4];
    std::uint32_t key[2];

    std::int32_t part = static_cast<std::int32_t>(state.part);
    if (part != 0) {
        res = state.result[3 - (--part)];
        skip_ahead(state, 1);
        return res;
    }
    counter[0] = state.counter[0];
    counter[1] = state.counter[1];
    counter[2] = state.counter[2];
    counter[3] = state.counter[3];
    key[0] = state.key[0];
    key[1] = state.key[1];

    round_10(counter, key);

    res = counter[0];

    skip_ahead(state, 1);
    return res;
}

} // namespace philox4x32x10_impl

template <>
class engine_base<oneapi::mkl::rng::device::philox4x32x10> {
protected:
    engine_base(std::uint64_t seed, std::uint64_t offset = 0) {
        philox4x32x10_impl::init(this->state_, 1, &seed, offset);
    }

    engine_base(std::uint64_t n, const std::uint64_t* seed, std::uint64_t offset = 0) {
        philox4x32x10_impl::init(this->state_, n, seed, offset);
    }

    engine_base(std::uint64_t seed, std::uint64_t n_offset, const std::uint64_t* offset_ptr) {
        philox4x32x10_impl::init(this->state_, 1, &seed, n_offset, offset_ptr);
    }

    engine_base(std::uint64_t n, const std::uint64_t* seed, std::uint64_t n_offset,
                const std::uint64_t* offset_ptr) {
        philox4x32x10_impl::init(this->state_, n, seed, n_offset, offset_ptr);
    }

    template <typename RealType>
    RealType generate(RealType a, RealType b) {
        RealType res;
        std::uint32_t res_uint;
        RealType a1;
        RealType c1;

        c1 = (b - a) / (static_cast<RealType>((std::numeric_limits<std::uint32_t>::max)()) + 1);
        a1 = (b + a) / static_cast<RealType>(2.0);

        res_uint = philox4x32x10_impl::generate_single(this->state_);
        res = static_cast<RealType>(static_cast<std::int32_t>(res_uint)) * c1 + a1;
        return res;
    }

    std::uint32_t generate() {
        return philox4x32x10_impl::generate_single(this->state_);
    }

    template <typename RealType>
    RealType generate_single(RealType a, RealType b) {
        return generate(a, b);
    }

    std::uint32_t generate_single() {
        return generate();
    }

    void skip_ahead(std::uint64_t num_to_skip) {
        detail::philox4x32x10_impl::skip_ahead(this->state_, num_to_skip);
    }

    void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) {
        detail::philox4x32x10_impl::skip_ahead(this->state_, num_to_skip.size(),
                                               num_to_skip.begin());
    }

    engine_state<oneapi::mkl::rng::device::philox4x32x10> state_;
};

// oneMKL RNG Uniform implementation

template <typename Type, typename Method>
class distribution_base<oneapi::mkl::rng::device::uniform<Type, Method>> {
public:
    struct param_type {
        param_type(Type a, Type b) : a_(a), b_(b) {}
        Type a_;
        Type b_;
    };

    distribution_base(Type a, Type b) : a_(a), b_(b) {
#ifndef __SYCL_DEVICE_ONLY__
        if (a >= b) {
            throw std::invalid_argument("uniform distribution: parameter 'a' must be less than 'b'");
        }
#endif
    }

    Type a() const {
        return a_;
    }

    Type b() const {
        return b_;
    }

    param_type param() const {
        return param_type(a_, b_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.a_ >= pt.b_) {
            throw std::invalid_argument("uniform distribution: parameter 'a' must be less than 'b'");
        }
#endif
        a_ = pt.a_;
        b_ = pt.b_;
    }

protected:
    template <typename EngineType>
    Type generate(EngineType& engine) {
        using FpType = typename std::conditional<!std::is_same_v<Method, uniform_method::accurate>, float, double>::type;
        Type res;
        if constexpr (std::is_integral<Type>::value) {
            FpType res_fp = engine.generate(static_cast<FpType>(a_), static_cast<FpType>(b_));
            res_fp = std::floor(res_fp);
            res = static_cast<Type>(res_fp);
            return res;
        }
        else {
            res = engine.generate(a_, b_);
            if constexpr (std::is_same<Method, uniform_method::accurate>::value) {
                res = std::max(res, a_);
                res = std::min(res, b_);
            }
        }
        return res;
    }

    template <typename EngineType>
    Type generate_single(EngineType& engine) {
        using FpType = typename std::conditional<
            !std::is_same_v<Method, uniform_method::accurate> ||
                std::is_same_v<Type, std::int8_t> || std::is_same_v<Type, std::uint8_t> ||
                std::is_same_v<Type, std::int16_t> || std::is_same_v<Type, std::uint16_t>,
            float, double>::type;
        Type res;
        if constexpr (std::is_integral<Type>::value) {
            FpType res_fp =
                engine.generate_single(static_cast<FpType>(a_), static_cast<FpType>(b_));
            res_fp = std::floor(res_fp);
            res = static_cast<Type>(res_fp);
            return res;
        }
        else {
            res = engine.generate_single(a_, b_);
            if constexpr (std::is_same<Method, uniform_method::accurate>::value) {
                res = std::max(res, a_);
                res = std::min(res, b_);
            }
        }
        return res;
    }

    Type a_;
    Type b_;
};

template <typename IntType, typename Method>
class distribution_base<oneapi::mkl::rng::device::bernoulli<IntType, Method>> {
public:
    struct param_type {
        param_type(float p) : p_(p) {}
        float p_;
    };

    distribution_base(float p) : p_(p) {
#ifndef __SYCL_DEVICE_ONLY__
        if ((p > 1.0f) || (p < 0.0f)) {
            throw std::invalid_argument("bernoulli distribution: parameter 'p' must be in range [0, 1]");
        }
#endif
    }

    float p() const {
        return p_;
    }

    param_type param() const {
        return param_type(p_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if ((pt.p_ > 1.0f) || (pt.p_ < 0.0f)) {
            throw std::invalid_argument("bernoulli distribution: parameter 'p' must be in range [0, 1]");
        }
#endif
        p_ = pt.p_;
    }

protected:
    template <typename EngineType>
    IntType generate(EngineType& engine) {
        auto uni_res = engine.generate(0.0f, 1.0f);
        return IntType{ uni_res < p_ };
    }

    template <typename EngineType>
    IntType generate_single(EngineType& engine) {
        auto uni_res = engine.generate_single(0.0f, 1.0f);
        return IntType{ uni_res < p_ };
    }

    float p_;
};

} // namespace detail

// Class template oneapi::mkl::rng::device::philox4x32x10
//
// Represents Philox4x32-10 counter-based pseudorandom number generator
//
// Supported parallelization methods:
//      skip_ahead
//
class philox4x32x10 : public detail::engine_base<philox4x32x10> {
public:
    static constexpr std::uint64_t default_seed = 0;

    philox4x32x10() : detail::engine_base<philox4x32x10>(default_seed) {}

    philox4x32x10(std::uint64_t seed, std::uint64_t offset = 0)
            : detail::engine_base<philox4x32x10>(seed, offset) {}

    philox4x32x10(std::initializer_list<std::uint64_t> seed, std::uint64_t offset = 0)
            : detail::engine_base<philox4x32x10>(seed.size(), seed.begin(), offset) {}

    philox4x32x10(std::uint64_t seed, std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<philox4x32x10>(seed, offset.size(), offset.begin()) {}

    philox4x32x10(std::initializer_list<std::uint64_t> seed,
                  std::initializer_list<std::uint64_t> offset)
            : detail::engine_base<philox4x32x10>(seed.size(), seed.begin(), offset.size(),
                                                          offset.begin()) {}

private:
    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::uint64_t num_to_skip);

    template <typename Engine>
    friend void skip_ahead(Engine& engine, std::initializer_list<std::uint64_t> num_to_skip);

    template <typename DistrType>
    friend class detail::distribution_base;
};

} // namespace oneapi::mkl::rng::device

#endif // _MKL_RNG_DEVICE_PHILOX4X32X10_UNIFORM_BERNOULLI_IMPL_HPP_
