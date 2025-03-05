from networks.modules import *
from utils.word_vectorizer import POS_enumerator
from os.path import join as pjoin

def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):
    def __init__(self, opt):
        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.projection = nn.Linear(512, 1024).to(opt.device)  # Проекционный слой
        
        self.opt = opt
        self.device = opt.device

        # Перенос всех компонентов на устройство
        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)
        
        # Режим оценки
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
        self.projection.eval()

    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            # Перенос данных на устройство
            device = self.device
            word_embs = word_embs.to(device).float()
            pos_ohot = pos_ohot.to(device).float()
            motions = motions.to(device).float()
            m_lens = m_lens.to(device)
            cap_lens = cap_lens.to(device)

            # 1. Сортировка по длинам движений (m_lens)
            sorted_m_lens, sort_idx = torch.sort(m_lens, descending=True)
            
            # 2. Применяем сортировку ко всем данным
            motions = motions[sort_idx]
            word_embs = word_embs[sort_idx]
            pos_ohot = pos_ohot[sort_idx]
            cap_lens = cap_lens[sort_idx]

            # 3. Кодирование движений
            movements = self.movement_encoder(motions[..., :-4])
            movements = self.projection(movements)
            
            # 4. Корректировка длин для GRU
            adjusted_m_lens = (sorted_m_lens // self.opt.unit_length).clamp(min=1)
            
            # 5. Упаковка последовательностей движений
            packed_motions = torch.nn.utils.rnn.pack_padded_sequence(
                movements,
                lengths=adjusted_m_lens.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            
            # 6. Кодирование движений
            self.motion_encoder.gru.flatten_parameters()
            _, motion_hidden = self.motion_encoder.gru(packed_motions)
            motion_embedding = motion_hidden[-1]

            # 7. Упаковка текстовых данных с enforce_sorted=False
            packed_text = torch.nn.utils.rnn.pack_padded_sequence(
                torch.cat([word_embs, pos_ohot], dim=-1),
                lengths=cap_lens.cpu(),
                batch_first=True,
                enforce_sorted=False  # Разрешаем неотсортированные длины
            )
            
            # 8. Кодирование текста
            _, text_hidden = self.text_encoder.gru(packed_text)
            text_embedding = text_hidden[-1]

            # 9. Восстановление исходного порядка
            reverse_idx = torch.argsort(sort_idx)
            return text_embedding[reverse_idx], motion_embedding[reverse_idx]

    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            # Перенос данных на устройство
            motions = motions.to(self.device).float()
            m_lens = m_lens.to(self.device).long()

            # Кодирование с проекцией
            movements = self.movement_encoder(motions[..., :-4])
            movements = self.projection(movements)  # 512 -> 1024

            # Корректировка длин
            adjusted_m_lens = (m_lens // self.opt.unit_length).clamp(min=1)
            
            # Упаковка и GRU
            sorted_lens, sort_idx = torch.sort(adjusted_m_lens, descending=True)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                movements[sort_idx],
                lengths=sorted_lens.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            
            self.motion_encoder.gru.flatten_parameters()
            _, hidden = self.motion_encoder.gru(packed)
            
            # Восстановление порядка
            _, reverse_idx = torch.sort(sort_idx)
            return hidden[-1][reverse_idx]
