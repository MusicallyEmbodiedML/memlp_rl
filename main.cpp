#include <iostream>
#include <memory>

#include "memlp/MLP.h"
#include "memlp/ReplayMemory.hpp"

#include "TestEnv.h"
#include "OrnsteinUhleneckNoise.h"

#include <chrono>

//easy RL scenario 1: set freq of pulse generator to specific value
//1 continuous action: frequency
//reward = distance from target frequency

constexpr size_t nInputs=1;
constexpr size_t bias=1;
constexpr size_t nOutputs = 1;

static const std::vector<ACTIVATION_FUNCTIONS> layers_activfuncs = {
    RELU, RELU, TANH
};

const std::vector<size_t> layers_nodes = {
    nInputs + bias,
    10, 10,
    nOutputs
};

const bool use_constant_weight_init = false;
const float constant_weight_init = 0;


void actorCriticTest() {
    std::shared_ptr<MLP<float> > actor, actorTarget, critic, criticTarget;

    const float discountFactor = 0.95;
    const float learningRate = 0.005;
    const float smoothingAlpha = 0.005;

    double theta = 0.15;  // Reversion speed
    double mu = 0.0;      // Long-term mean
    double sigma = 0.3;   // Noise intensity
    double dt = 0.01;     // Time step

    OrnsteinUhlenbeckNoise ou_noise(theta, mu, sigma, dt);
    const size_t stateSize = 1;
    const size_t actionSize = 1;
    CustomEnv env(stateSize, actionSize, 999999999);
    auto currentState = env.reset();

    //init networks
    actor = std::make_shared<MLP<float> > (
        layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );
    actorTarget = std::make_shared<MLP<float> > (
        layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );

    const std::vector<size_t> critic_layers_nodes = {
        stateSize + actionSize + bias,
        10, 10,
        1
    };

    critic = std::make_shared<MLP<float> > (
        critic_layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );
    criticTarget = std::make_shared<MLP<float> > (
        critic_layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );

    struct trainRLItem {
        std::vector<float> state ;
        std::vector<float> action;
        float reward;
        std::vector<float> nextState;
    };

    ReplayMemory<trainRLItem> replayMem;

    std::vector<float> actorOutput, criticOutput;
    std::vector<float> criticInput(critic_layers_nodes[0]);

    for(size_t step=0; step < 30; step++) {
        std::cout << "step: " << step << std::endl;
        //get action from actor
        actor->GetOutput({currentState[0], 1.f}, &actorOutput);
        //add noise
        float noise = ou_noise.sample();
        actorOutput[0] += noise;
        std::cout << "action: " << actorOutput[0] << ", N: " << noise << std::endl;

        //perform the action
        auto [next_state, reward, done, info] = env.step(actorOutput);
        // Print results
        std::cout << "Next state: ";
        for (float s : next_state) std::cout << s << " ";
        std::cout << "; Reward: " << reward << std::endl;

        //add to replay mem
        trainRLItem trainItem = {currentState, actorOutput, reward, next_state};
        replayMem.add(trainItem, step);

        std::vector<trainRLItem> sample = replayMem.sample(4);

        //run sample through critic target, build training set for critic net
        MLP<float>::training_pair_t ts;
        for(size_t i = 0; i < sample.size(); i++) {
            //---calculate y
            //--calc next-state-action pair
            //get next action from actorTarget given next state
            auto nextStateInput =  sample[i].nextState;
            nextStateInput.push_back(1.f); // bias
            actorTarget->GetOutput(nextStateInput, &actorOutput);
            //use criticTarget to estimate value of next action given next state
            criticInput[0] = sample[i].nextState[0];
            criticInput[1] = actorOutput[0];
            criticInput[2] = 1.f; //bias
            criticTarget->GetOutput(criticInput, &criticOutput);

            //calculate expected reward
            float y = reward + (discountFactor *  criticOutput[0]);
            std::cout << "[" << i << "]: y: " << y << std::endl;

            criticInput[0] = sample[i].state[0];
            criticInput[1] = sample[i].action[0];
            criticInput[2] = 1.f; //bias
            ts.first.push_back(criticInput);
            ts.second.push_back({y});
        }
        critic->Train(ts, learningRate, 1);

        //update the actor

        criticTarget->SmoothUpdateWeights(critic, smoothingAlpha);
        actorTarget->SmoothUpdateWeights(actor, smoothingAlpha);
    }

}

void standardTest() {
    std::shared_ptr<MLP<float> > net;

    std::cout << "Standard test" << std::endl;
    net = std::make_shared<MLP<float> > (
        layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );

    //train
    MLP<float>::training_pair_t ts;
    for(size_t i=0; i < 1000; i++) {
        ts.first.push_back({i/1000.f,1.f});
        ts.second.push_back({i < 500 ? -1.f : 0.5f});
    }
    net->Train(ts, 0.1, 1000, 0.00001, true);

    //test
    std::vector<float> output;
    for(float i=0.f; i < 1.f; i += 0.05f) {
        net->GetOutput({i,1.f}, &output);
        std::cout << i << ": " << output[0] << std::endl;
    }
}
void printVector(std::vector<float> &v) {
    for(size_t i=0; i < v.size(); i++) {
        std::cout << v[i] << " ";
    }
}
void replayMemTest() {
    std::shared_ptr<MLP<float> > net;

    std::cout << "Replay mem test" << std::endl;
    net = std::make_shared<MLP<float> > (
        layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );


    //train loop
    MLP<float>::training_pair_t ts;
    ReplayMemory<trainXYItem<float> > replayMem;

    for(size_t i=0; i < 10000; i++) {
        float x = rand() / (float) RAND_MAX;
        trainXYItem<float> trainItem {{x,1.f}, {x > 0.5 ? 1.f : -1.f}};
        replayMem.add(trainItem, i);
        std::vector<trainXYItem<float> > sample = replayMem.sample(8);
        std::cout << "iteration: " << i << std::endl;
        MLP<float>::training_pair_t tp;
        for(size_t j=0; j < sample.size(); j++) {
            std::cout << "[" << j << "]: ";
            printVector(sample[j].X);
            std::cout << " :: ";
            printVector(sample[j].Y);
            std::cout << std::endl;
            tp.first.push_back(sample[j].X);
            tp.second.push_back(sample[j].Y);
        }
        net->Train(tp, 0.05, 1, 0.00001, true);
    }

    //test
    std::vector<float> output;
    for(float i=0.f; i < 1.f; i += 0.05f) {
        net->GetOutput({i,1.f}, &output);
        std::cout << i << ": " << output[0] << std::endl;
    }
}



void actorEnvTest() {
    std::shared_ptr<MLP<float> > actor;

    CustomEnv env(1, 1, 999999999);

    for(int i=0; i < 100; i++) {
        auto res = env.step({i/100.f});
        std::cout << "Reward: " << std::get<0>(res)[0] << std::endl;
    }
}

void targetSmoothNetworkTest() {
    //learn one network, smooth update to identical one
    //grab predictions from each
    //target network should give smoothed out version of org network
    //probably need a challenging / nonlinear task
    std::shared_ptr<MLP<float> > net1;
    std::shared_ptr<MLP<float> > net2;


    std::cout << "Smooth network test" << std::endl;
    net1 = std::make_shared<MLP<float> > (
        layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );

    net2 = std::make_shared<MLP<float> > (
        layers_nodes,
        layers_activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );

    // loop
    MLP<float>::training_pair_t ts;
    ReplayMemory<trainXYItem<float> > replayMem;
    std::vector<float> losses;

    auto target = [](float val) {
        float targ;
        if (val < -0.3) {
            targ = 0.2;
        }else if (val < 0.3) {
            targ = -1;
        }else {
            targ = 1;
        }
        return targ;
    };

    std::vector<float> error1, error2;

    for(size_t i=0; i < 10000; i++) {
        float x = (rand() / (float) RAND_MAX * 2.f) - 1.f;
        trainXYItem<float> trainItem {{x,1.f}, {target(x)}};
        replayMem.add(trainItem, i);
        std::vector<trainXYItem<float> > sample = replayMem.sample(8);
        std::cout << "iteration: " << i << std::endl;
        MLP<float>::training_pair_t tp;
        for(size_t j=0; j < sample.size(); j++) {
        //     std::cout << "[" << j << "]: ";
        //     printVector(sample[j].X);
        //     std::cout << " :: ";
        //     printVector(sample[j].Y);
        //     std::cout << std::endl;
            tp.first.push_back(sample[j].X);
            tp.second.push_back(sample[j].Y);
        }
        float loss = net1->Train(tp, 0.005, 1, 0.00001, true);
        losses.push_back(loss);

        net2->SmoothUpdateWeights(net1, 0.005);

        //test
        std::vector<float> output;
        float error = 0;
        for(float i=-1.f; i < 1.f; i += 0.1f) {
            net1->GetOutput({i,1.f}, &output);
            error += fabs(output[0] - target(i));
            // std::cout << i << ": " << output[0] << std::endl;
        }
        error1.push_back(error);

        float err2 = 0;
        for(float i=-1.f; i < 1.f; i += 0.1f) {
            net2->GetOutput({i,1.f}, &output);
            err2 += fabs(output[0] - target(i));
            // std::cout << i << ": " << output[0] << std::endl;
        }
        error2.push_back(err2);
    }


    FILE* gnuplot = popen("gnuplot -persistent", "w");
    if (!gnuplot) {
        std::cerr << "Error opening gnuplot\n";
    }

    // fprintf(gnuplot, "plot '-' u 1:2 t 'data1' w lp lt 0, '-' u 1:2 t 'data2' lt 2\n");
    fprintf(gnuplot, "plot '-' u 1:2 t 'Series 1' w l lt 1, '-' u 1:2 t 'Series 2' w l lt 2\n");
    for (int i = 0; i < error1.size(); i++) {
        fprintf(gnuplot, "%f %f\n", (float)i, error1[i]);
        // std::cout << losses[i] << std::endl;

    }
    fprintf(gnuplot, "e\n");

    fprintf(gnuplot, "plot '-' with lines\n");
    for (int i = 0; i < error1.size(); i++) {
        fprintf(gnuplot, "%f %f\n", (float)i, error2[i]+1.f);

    }
    fprintf(gnuplot, "e\n");

    pclose(gnuplot);
}

void updateIdentityTest() {
    srand(time(NULL));
    std::cout << "CalcGradients test\n";
    //make a random network with single input and output, calc gradient wrt input using identify function as error
    //check if this is equiv to numerical differentiation?

    static const std::vector<ACTIVATION_FUNCTIONS> activfuncs = {
        RELU, RELU, TANH
    };

    const std::vector<size_t> nodes = {
        2+1,
        10, 10,
        1
    };

    std::shared_ptr<MLP<float> > net1;
    net1 = std::make_shared<MLP<float> > (
        nodes,
        activfuncs,
        loss::LOSS_MSE,
        use_constant_weight_init,
        constant_weight_init
    );

    std::cout << "Untrained grads:\n";
    std::vector<float> testInput = {1,1,1};
    std::vector<float> testerr = {1};

    net1->CalcGradients(testInput, testerr);
    std::vector<float> l0Grads = net1->m_layers[0].GetGrads();
    for(size_t i=0; i < l0Grads.size(); i++) {
        std::cout << l0Grads[i] << "\t";
    }
    std::cout << std::endl;

    MLP<float>::training_pair_t ts;
    ts.first.push_back({0,0,1});
    ts.second.push_back({0});
    ts.first.push_back({1,1,1});
    ts.second.push_back({1});


    size_t trials = 10;
    net1->Train(ts, 0.1, 10000, 0.0000000, true);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> output;
    std::vector<float> output2;
    for(size_t i=0; i < trials; i++) {
        net1->GetOutput(testInput, &output);
        net1->GetOutput({1+1e-4,1+1e-4,1+1e-4}, &output2);
    }
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate elapsed time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Output elapsed time
    std::cout << "Elapsed time (numerical diff): " << duration.count() << " ms" << std::endl;

    // std::cout << output[0] << std::endl;
    // std::cout << output2[0] << std::endl;
    std::cout << "Numerical diff: ";
    std::cout << (output2[0] - output[0])/1e-4 << std::endl;


    std::vector<float> output0;
    // net1->GetOutput({1,1,1}, &output0);
    std::vector<float> err = {1};

    start = std::chrono::high_resolution_clock::now();

    float gsum=0;
    for(size_t i=0; i < trials; i++) {
        net1->CalcGradients(testInput, err);
        auto grads = net1->m_layers[0].GetGrads();
        gsum = 0;
        for(size_t i=0; i < grads.size(); i++) {
            // std::cout << grads[i] << "\t";
            gsum += grads[i];
        }
        // std::cout << std::endl;
    }
    end = std::chrono::high_resolution_clock::now();
    // Calculate elapsed time in milliseconds
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Output elapsed time
    std::cout << "Elapsed time (autograd): " << duration.count() << " ms" << std::endl;

    std::cout << "Autograd: " << gsum << std::endl;
    // for(auto &v : ts.first) {
    //     // net1->ClearGradients();
    //
    //     net1->CalcGradients(v, err);
    //     std::vector<float> grads = net1->m_layers[0].GetGrads();
    //     for(size_t i=0; i < grads.size(); i++) {
    //         std::cout << grads[i] << "\t";
    //     }
    //     std::cout << std::endl;
    //
    // }


}

int main()
{
    std::cout << "MEMLP RL Test" << std::endl;
    // actorCriticTest();
    updateIdentityTest();
    return 0;
}
