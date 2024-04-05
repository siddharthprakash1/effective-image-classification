from manim import *
import numpy as np
import json

class TrainValAccuracyGraph(Scene):
    def construct(self):
        # Load the history data
        with open('training_history.json', 'r') as f:
            history_data = json.load(f)

        train_acc = np.array(history_data['accuracy'])
        val_acc = np.array(history_data['val_accuracy'])
        train_loss = np.array(history_data['loss'])
        val_loss = np.array(history_data['val_loss'])

        # Normalize data for better visualization
        max_acc = max(train_acc.max(), val_acc.max())
        max_loss = max(train_loss.max(), val_loss.max())
        train_acc /= max_acc
        val_acc /= max_acc
        train_loss /= max_loss
        val_loss /= max_loss

        # Create Axes for Accuracy and Loss
        acc_axes = Axes(x_range=[0, len(train_acc), 5], y_range=[0, 1.1, 0.2], x_length=6, y_length=3,
                        tips=False, axis_config={"color": BLUE})
        loss_axes = Axes(x_range=[0, len(train_loss), 5], y_range=[0, 1.1, 0.2], x_length=6, y_length=3,
                         tips=False, axis_config={"color": BLUE})

        acc_axes.to_edge(UP, buff=0.5)
        loss_axes.to_edge(DOWN, buff=0.5)

        # Labels
        acc_title = Text("Training and Validation Accuracy").scale(0.5).move_to(acc_axes.get_top() + UP*0.5)
        loss_title = Text("Training and Validation Loss").scale(0.5).move_to(loss_axes.get_top() + UP*0.5)

        # Plotting
        epochs = np.arange(1, len(train_acc) + 1)
        train_acc_graph = acc_axes.plot_line_graph(x_values=epochs, y_values=train_acc, add_vertex_dots=False,
                                                    line_color=GREEN, stroke_width=4)
        val_acc_graph = acc_axes.plot_line_graph(x_values=epochs, y_values=val_acc, add_vertex_dots=False,
                                                  line_color=RED, stroke_width=4)
        train_loss_graph = loss_axes.plot_line_graph(x_values=epochs, y_values=train_loss, add_vertex_dots=False,
                                                      line_color=GREEN, stroke_width=4, legend="Training")
        val_loss_graph = loss_axes.plot_line_graph(x_values=epochs, y_values=val_loss, add_vertex_dots=False,
                                                    line_color=RED, stroke_width=4, legend="Validation")

        # Display
        self.play(Create(acc_axes), Create(loss_axes), Write(acc_title), Write(loss_title))
        self.play(Create(train_acc_graph), Create(val_acc_graph))
        self.play(Create(train_loss_graph), Create(val_loss_graph))
        self.wait(1)
