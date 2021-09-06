import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import music21 as m2
import pickle
from params import *
import crash

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.RNN(INPUT_DIM, HIDDEN_DIM, N_LAYERS, batch_first=True)
        
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [batch size, seq len, input_dim]
                
        outputs, hidden = self.rnn(src)
        
        #outputs = [batch size, seq len, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.output_dim = output_dim
        # self.hid_dim = hid_dim
        # self.n_layers = n_layers
        
        # self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.RNN(OUTPUT_DIM, HIDDEN_DIM, N_LAYERS, batch_first=True)
        
        self.fc_out = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        
        # self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden):
        
        #x = [batch size, output_dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        x = x.unsqueeze(1)
        
        #x = [batch size, 1, output_dim]
        
        # embedded = self.dropout(self.embedding(x))
        #embedded = [1, batch size, emb dim]
                
        output, hidden = self.rnn(x, hidden)
        
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg = None, teacher_forcing_ratio = 0.5):
        
        #src = [batch size, seq len, input_dim]
        #trg = [batch size, seq len, output_dim]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # the actual 
        seq_len = src.shape[1]

        #tensor to store decoder outputs
        outputs = torch.zeros(src.shape[0], seq_len, OUTPUT_DIM)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src)

        #first input to the decoder is the <sos> tokens
        if trg == None: # the inference case
            trg = torch.zeros(src.shape[0], seq_len, OUTPUT_DIM)
            teacher_forcing_ratio = 0
        x = trg[:, 0, :]
        
        for t in range(1, seq_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(x, hidden)
            
            #place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output.squeeze(1)
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            # top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            x = trg[:, t, :] if teacher_force else output.squeeze(1)

        return outputs

def load_data():
    with open("assets/chords_lists_hooktheory.txt", "r") as f:
        all_melody, all_harmony = [], []
        for line in f.read().splitlines():
            melody, harmony = line.split("|")

            # [0, 7, 4]
            melody = [PITCHES.index(m) for m in melody.split()]
            try:
                harmony = [CHORDS.index(chord) for chord in harmony.split(" ")]

                while len(melody) >= SEQ_LEN:
                    # [0, 19, 11]
                    all_harmony.append(harmony[:SEQ_LEN])
                    all_melody.append(melody[:SEQ_LEN])
                    harmony = harmony[SEQ_LEN:]
                    melody = melody[SEQ_LEN:]
            except:
                pass

        all_melody = F.one_hot(torch.tensor(all_melody), num_classes=INPUT_DIM).float()
        all_harmony = F.one_hot(torch.tensor(all_harmony), num_classes=OUTPUT_DIM).float()

    return all_melody, all_harmony

def train_rnn(all_melody, all_harmony):
        
    N = int(len(all_harmony) * 0.8)

    enc = Encoder()
    dec = Decoder()
    model = Seq2Seq(enc, dec)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)


    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()

        for i in range(0, N - BATCH_SIZE, BATCH_SIZE):
            # melody: (batch_size, seq_len, input_dim)
            # harmony: (batch_size, seq_len, output_dim)
            melody, harmony = all_melody[i:i+BATCH_SIZE], all_harmony[i:i+BATCH_SIZE]

            outputs = model(melody, trg=harmony)
            loss = criterion(outputs.view(-1, OUTPUT_DIM), harmony.argmax(2).view(-1))
            loss.backward() 
            optimizer.step() 

        print('Epoch: {}/{}.............'.format(epoch, N_EPOCHS), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

    # torch.save(model, "assets/rnnmodel.pt")

    torch.save({
                'model_state_dict': model.state_dict()
                }, "assets/rnnmodel.pt")

    return model

def sonify(melody, harmony):

    s = m2.stream.Stream()

    for i in range(SEQ_LEN):
        s.insert(i, m2.note.Note(PITCHES[melody[i]]))
        s.insert(i, m2.chord.Chord(
            m2.harmony.ChordSymbol(CHORDS2[harmony[i]]).pitches))

    mf = m2.midi.translate.streamToMidiFile(s)
    mf.open("test.mid", "wb")
    mf.write()
    mf.close()

    return

if __name__ == '__main__':

    all_melody, all_harmony = load_data()
    # model = train_rnn(all_melody, all_harmony)

    model = Seq2Seq(Encoder(), Decoder())
    model.load_state_dict(torch.load("assets/rnnmodel.pt")['model_state_dict'])
    model.eval()

    # i = int(len(all_harmony) * 0.8)
    i = int(random.random() * 200)
    print(i)
    melody = all_melody[i]
    outputs = model(melody.unsqueeze(0))
    harmony = outputs.argmax(2).squeeze(0)
    # sonify(melody.argmax(1), all_harmony[i].argmax(1))
    sonify(melody.argmax(1), harmony)
    # crash()








