#!/usr/bin/env python3
# Copyright 2025 Upbeat, Inc
# SPDX-License-Identifier: Apache-2.0

"""
Static Memory Allocation Optimizer for UPDL Runtime

This module performs static analysis to compute minimal memory allocation
for neural network graph execution, optimized for embedded systems.
"""

import json
import os
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from .logger import log_info, log_debug, log_error


@dataclass
class LayerNode:
    """Represents a layer in the execution graph"""
    index: int
    name: str
    layer_type: str
    input_shape: List[int]
    output_shape: List[int]
    input_layers: List[int]  # which layers provide inputs
    consumer_layers: List[int]  # which layers consume this output
    execution_order: int = -1  # topological order for execution


@dataclass
class BufferAllocation:
    """Represents a memory buffer allocation"""
    buffer_id: int
    size_bytes: int
    alignment: int = 4
    assigned_layers: List[int] = None

    def __post_init__(self):
        if self.assigned_layers is None:
            self.assigned_layers = []


class MemoryPlanner:
    """
    Static memory allocation optimizer using graph coloring algorithm.

    Performs liveness analysis and optimal buffer allocation to minimize
    memory footprint for neural network graph execution.
    """

    def __init__(self, element_size: int = 2):  # int16_t = 2 bytes
        self.element_size = element_size
        self.layers: Dict[int, LayerNode] = {}
        self.execution_order: List[int] = []

    def add_layer(self, index: int, name: str, layer_type: str,
                  input_shape: List[int], output_shape: List[int],
                  input_layers: List[int] = None) -> None:
        """Add a layer to the graph for analysis"""
        if input_layers is None:
            input_layers = [index - 1] if index > 0 else []

        # Clean shapes - convert None to reasonable defaults
        clean_input_shape = self._clean_shape(input_shape, default_val=1)
        clean_output_shape = self._clean_shape(output_shape, default_val=1)

        layer = LayerNode(
            index=index,
            name=name,
            layer_type=layer_type,
            input_shape=clean_input_shape,
            output_shape=clean_output_shape,
            input_layers=input_layers,
            consumer_layers=[]
        )

        self.layers[index] = layer

        # Update consumer information for input layers
        for input_idx in input_layers:
            if input_idx in self.layers:
                self.layers[input_idx].consumer_layers.append(index)

        log_debug(f"Added layer {index}: {name} ({layer_type}) - shapes: {clean_input_shape} -> {clean_output_shape}")

    def _clean_shape(self, shape: List, default_val: int = 1) -> List[int]:
        """Clean shape list by replacing None values with defaults"""
        if not shape:
            return [default_val]

        cleaned = []
        for dim in shape:
            if dim is None:
                cleaned.append(default_val)
            elif isinstance(dim, (int, float)) and dim > 0:
                cleaned.append(int(dim))
            else:
                cleaned.append(default_val)

        return cleaned

    def compute_execution_order(self) -> List[int]:
        """
        Compute topological execution order using Kahn's algorithm.

        Returns:
            List of layer indices in valid execution order
        """
        # Build dependency graph
        in_degree = {idx: len(layer.input_layers) for idx, layer in self.layers.items()}
        queue = [idx for idx, degree in in_degree.items() if degree == 0]
        execution_order = []

        while queue:
            current = queue.pop(0)
            execution_order.append(current)

            # Process consumers
            for consumer_idx in self.layers[current].consumer_layers:
                in_degree[consumer_idx] -= 1
                if in_degree[consumer_idx] == 0:
                    queue.append(consumer_idx)

        if len(execution_order) != len(self.layers):
            raise ValueError("Cyclic dependency detected in graph")

        # Update execution order in layer nodes
        for order, layer_idx in enumerate(execution_order):
            self.layers[layer_idx].execution_order = order

        self.execution_order = execution_order
        log_info(f"Computed execution order: {execution_order}")
        return execution_order

    def analyze_liveness(self) -> Dict[int, Tuple[int, int]]:
        """
        Analyze liveness intervals for each layer's output.

        Returns:
            Dict mapping layer_idx -> (birth_time, death_time)
            where times are based on execution order
        """
        if not self.execution_order:
            self.compute_execution_order()

        liveness = {}

        for layer_idx, layer in self.layers.items():
            birth_time = layer.execution_order

            # Death time is when the last consumer executes
            if layer.consumer_layers:
                death_times = [self.layers[consumer].execution_order
                              for consumer in layer.consumer_layers]
                death_time = max(death_times)
            else:
                # Output layer - lives until end
                death_time = len(self.layers) - 1

            liveness[layer_idx] = (birth_time, death_time)
            log_debug(f"Layer {layer_idx} liveness: [{birth_time}, {death_time}]")

        return liveness

    def build_interference_graph(self) -> Dict[int, Set[int]]:
        """
        Build interference graph where edges connect layers with overlapping lifetimes.

        Returns:
            Adjacency list representation of interference graph
        """
        liveness = self.analyze_liveness()
        interference = {idx: set() for idx in self.layers.keys()}

        layer_indices = list(self.layers.keys())
        for i, layer_a in enumerate(layer_indices):
            for layer_b in layer_indices[i+1:]:
                birth_a, death_a = liveness[layer_a]
                birth_b, death_b = liveness[layer_b]

                # Check if lifetimes overlap
                if not (death_a < birth_b or death_b < birth_a):
                    interference[layer_a].add(layer_b)
                    interference[layer_b].add(layer_a)
                    log_debug(f"Interference: Layer {layer_a} <-> Layer {layer_b}")

        return interference

    def compute_buffer_sizes(self) -> Dict[int, int]:
        """Compute required buffer size for each layer output"""
        buffer_sizes = {}

        for layer_idx, layer in self.layers.items():
            # Calculate tensor size in bytes, handling None values
            if not layer.output_shape or any(dim is None for dim in layer.output_shape):
                log_error(f"Layer {layer_idx} has invalid output shape: {layer.output_shape}")
                # Use a default size for layers with invalid shapes
                size_elements = 1024  # Default fallback
            else:
                # Filter out None values and ensure all dimensions are positive
                valid_dims = [dim for dim in layer.output_shape if dim is not None and dim > 0]
                size_elements = np.prod(valid_dims) if valid_dims else 1

            size_bytes = size_elements * self.element_size

            # Apply 4-byte alignment for hardware compatibility
            aligned_size = ((size_bytes + 3) // 4) * 4
            buffer_sizes[layer_idx] = aligned_size

            log_debug(f"Layer {layer_idx} buffer size: {layer.output_shape} -> {size_elements} elements -> {aligned_size} bytes")

        return buffer_sizes

    def graph_coloring_allocation(self) -> Tuple[List[BufferAllocation], Dict[int, int], Dict[int, int]]:
        """
        Perform graph coloring to assign minimal buffers.

        Uses greedy coloring algorithm optimized for embedded constraints.
        """
        interference = self.build_interference_graph()
        buffer_sizes = self.compute_buffer_sizes()

        # Sort layers by degree (most constrained first) for better coloring
        layer_degrees = [(idx, len(neighbors)) for idx, neighbors in interference.items()]
        layer_degrees.sort(key=lambda x: x[1], reverse=True)

        # Color assignment: layer_idx -> buffer_id
        coloring = {}
        buffer_allocations = []

        for layer_idx, _ in layer_degrees:
            layer_size = buffer_sizes[layer_idx]

            # Find available color (buffer) that doesn't conflict
            used_colors = {coloring[neighbor] for neighbor in interference[layer_idx]
                          if neighbor in coloring}

            # Try to reuse existing buffer of sufficient size
            available_buffer = None
            for buffer_id, buffer in enumerate(buffer_allocations):
                if (buffer_id not in used_colors and
                    buffer.size_bytes >= layer_size):
                    available_buffer = buffer_id
                    break

            if available_buffer is not None:
                # Reuse existing buffer
                coloring[layer_idx] = available_buffer
                buffer_allocations[available_buffer].assigned_layers.append(layer_idx)
                log_debug(f"Layer {layer_idx} reuses buffer {available_buffer}")
            else:
                # Create new buffer
                new_buffer_id = len(buffer_allocations)
                coloring[layer_idx] = new_buffer_id

                new_buffer = BufferAllocation(
                    buffer_id=new_buffer_id,
                    size_bytes=layer_size,
                    assigned_layers=[layer_idx]
                )
                buffer_allocations.append(new_buffer)
                log_debug(f"Layer {layer_idx} gets new buffer {new_buffer_id} ({layer_size} bytes)")

        # Calculate memory savings
        total_naive = sum(buffer_sizes.values())
        total_optimized = sum(buf.size_bytes for buf in buffer_allocations)
        savings_pct = ((total_naive - total_optimized) / total_naive) * 100

        log_info(f"Buffers needed: {len(buffer_allocations)}")
        log_info(f"Optimized allocation: {total_optimized} bytes")

        return buffer_allocations, coloring, buffer_sizes

    def generate_allocation_metadata(self, log_path: Optional[str] = None) -> Dict:
        """
        Generate metadata for runtime buffer allocation.

        Returns:
            Dictionary containing buffer allocation information for serialization
        """
        buffer_allocations, coloring, buffer_sizes = self.graph_coloring_allocation()
        liveness = self.analyze_liveness()

        metadata = {
            "buffer_count": len(buffer_allocations),
            "total_memory_bytes": sum(buf.size_bytes for buf in buffer_allocations),
            "buffers": [],
            "layer_buffer_mapping": coloring
        }

        for buffer in buffer_allocations:
            buffer_info = {
                "buffer_id": buffer.buffer_id,
                "size_bytes": buffer.size_bytes,
                "alignment": buffer.alignment,
                "assigned_layers": buffer.assigned_layers
            }
            metadata["buffers"].append(buffer_info)

        if log_path:
            self._write_lifetime_log(
                log_path=log_path,
                buffer_allocations=buffer_allocations,
                coloring=coloring,
                buffer_sizes=buffer_sizes,
                liveness=liveness
            )

        return metadata

    def _write_lifetime_log(self, log_path: str,
                            buffer_allocations: List[BufferAllocation],
                            coloring: Dict[int, int],
                            buffer_sizes: Dict[int, int],
                            liveness: Dict[int, Tuple[int, int]]) -> None:
        """Persist allocation details for debugging."""
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_data = {
                "execution_order": self.execution_order,
                "buffers": [
                    {
                        "buffer_id": buf.buffer_id,
                        "size_bytes": buf.size_bytes,
                        "assigned_layers": buf.assigned_layers
                    }
                    for buf in buffer_allocations
                ],
                "layers": []
            }

            for layer_idx in sorted(self.layers.keys()):
                layer = self.layers[layer_idx]
                birth, death = liveness.get(layer_idx, (-1, -1))
                log_data["layers"].append(
                    {
                        "index": layer_idx,
                        "name": layer.name,
                        "type": layer.layer_type,
                        "input_layers": layer.input_layers,
                        "consumer_layers": layer.consumer_layers,
                        "output_shape": layer.output_shape,
                        "buffer_id": coloring.get(layer_idx),
                        "buffer_size_bytes": buffer_sizes.get(layer_idx),
                        "lifetime": {"birth": birth, "death": death},
                    }
                )

            with open(log_path, "w") as fp:
                json.dump(log_data, fp, indent=2, default=int)

        except Exception as exc:
            log_error(f"Memory Optimizer: Failed to write lifetime log ({exc})")


def optimize_memory_for_graph(layers_data: Dict, log_path: Optional[str] = None) -> Dict:
    """
    Main entry point for memory optimization.

    Args:
        layers_data: Dictionary of layer information from fused_data

    Returns:
        Memory allocation metadata for runtime
    """
    optimizer = MemoryPlanner()

    # Build graph from layers_data
    for layer_idx, (layer_name, layer_data) in enumerate(layers_data.items()):
        input_layers = layer_data.get("input_layer_indices", [layer_idx - 1] if layer_idx > 0 else [])

        # Get shapes and ensure they're not None
        input_shape = layer_data.get("input_shape", [1, 1, 1, 1])
        output_shape = layer_data.get("output_shape", [1, 1, 1, 1])

        # Handle nested shape structures
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0:
            if isinstance(input_shape[0], (tuple, list)):
                input_shape = list(input_shape[0])

        if isinstance(output_shape, (tuple, list)) and len(output_shape) > 0:
            if isinstance(output_shape[0], (tuple, list)):
                output_shape = list(output_shape[0])

        log_debug(f"Processing layer {layer_idx}: {layer_name} - input: {input_shape}, output: {output_shape}")

        optimizer.add_layer(
            index=layer_idx,
            name=layer_name,
            layer_type=layer_data.get("layer_type", "Unknown"),
            input_shape=input_shape,
            output_shape=output_shape,
            input_layers=input_layers
        )

    # Generate optimized allocation
    return optimizer.generate_allocation_metadata(log_path=log_path)
