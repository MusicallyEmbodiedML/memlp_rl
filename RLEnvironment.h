#ifndef RL_ENVIRONMENT_H
#define RL_ENVIRONMENT_H

#include <vector>
#include <utility>

class RLEnvironment {
public:
    // Virtual destructor for proper cleanup of derived classes
    virtual ~RLEnvironment() = default;

    // Reset the environment to its initial state and return the initial observation
    virtual std::vector<float> reset() = 0;

    // Step function: takes an action and returns the new state, reward, done flag, and optional info
    virtual std::tuple<std::vector<float>, float, bool, std::vector<float>> step(const std::vector<float>& action) = 0;

    // Get the action space size
    virtual int action_size() const = 0;

    // Get the observation space size
    virtual int state_size() const = 0;

    // Render function (optional, for visualization)
    virtual void render() {}

    // Close the environment (if necessary)
    virtual void close() {}
};

#endif // RL_ENVIRONMENT_H
