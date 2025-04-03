#include "RLEnvironment.h"
#include <iostream>
#include <cmath>

class CustomEnv : public RLEnvironment {
private:
    std::vector<float> state;
    int state_dim;
    int action_dim;
    int steps;
    int max_steps;

public:
    CustomEnv(int state_size, int action_size, int max_ep_steps)
        : state_dim(state_size), action_dim(action_size), max_steps(max_ep_steps), steps(0) {
        state.resize(state_dim, 0.0f);
    }

    std::vector<float> reset() override {
        state = std::vector<float>(state_dim, 0.0f);
        steps = 0;
        return state;
    }

    std::tuple<std::vector<float>, float, bool, std::vector<float>> step(const std::vector<float>& action) override {
        // Apply action (dummy logic for example)
        for (size_t i = 0; i < state.size(); i++) {
            state[i] += action[i] * 0.1f; // Small state update
        }

        // Reward function: simple negative distance from zero
        float reward = -std::fabs(state[0]);

        // Check if done
        bool done = (++steps >= max_steps);

        // Return next state, reward, done, and an empty info vector
        return {state, reward, done, {}};
    }

    int action_size() const override {
        return action_dim;
    }

    int state_size() const override {
        return state_dim;
    }

    void render() override {
        std::cout << "State: ";
        for (float s : state) std::cout << s << " ";
        std::cout << std::endl;
    }
};
