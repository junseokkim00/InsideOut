# python insideout.py --dataset_type eu \
#                     --model_type gpt-3.5-turbo-0125 \
#                     --num_round 2 \
#                     --summarize 1 \
#                     --select_emotions 0 \
#                     --add_rational 0


python insideout_chat.py --model_type gpt-3.5-turbo-0125 \
                    --num_round 2 \
                    --summarize 1 \
                    --select_emotions 1 \
                    --add_rational 1