"""
Batch Operations Commands.

Commands for running batch operations, scheduling tasks,
and managing automated workflows.
"""

import sys
import logging
import json
import os
import subprocess
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ..utils.formatting import format_table, format_timestamp, format_duration
from ..utils.errors import CliError, handle_cli_error

logger = logging.getLogger(__name__)


@click.group(name='batch')
def batch_group():
    """Batch operations and scheduling commands."""
    pass


@batch_group.command()
@click.argument('commands', nargs=-1, required=True)
@click.option('--parallel', is_flag=True, help='Run commands in parallel')
@click.option('--continue-on-error', is_flag=True, help='Continue if a command fails')
@click.option('--output-dir', type=click.Path(), help='Directory to save command outputs')
@click.option('--timeout', default=3600, help='Timeout for each command in seconds')
@click.pass_context
def run(ctx, commands: tuple, parallel: bool, continue_on_error: bool,
        output_dir: Optional[str], timeout: int):
    """Run multiple CLI commands in batch."""
    console = ctx.obj['console']

    try:
        console.print(f"[blue]Running batch of {len(commands)} commands...[/blue]")

        if parallel:
            console.print("[yellow]Parallel execution mode[/yellow]")
        else:
            console.print("[blue]Sequential execution mode[/blue]")

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            console.print(f"Output directory: {output_path}")

        console.print()

        # Prepare command execution
        results = []
        failed_commands = []

        if parallel:
            # Parallel execution using subprocess
            import concurrent.futures
            import threading

            def run_command(cmd_index, command):
                try:
                    start_time = datetime.utcnow()

                    # Prepare command
                    cmd_parts = command.split()
                    if cmd_parts[0] != 'finlab-cli':
                        cmd_parts = ['finlab-cli'] + cmd_parts

                    # Add common options from current context
                    if ctx.obj.get('verbose'):
                        cmd_parts.extend(['-v'] * ctx.obj['verbose'])
                    if ctx.obj.get('quiet'):
                        cmd_parts.append('-q')
                    if ctx.obj.get('no_color'):
                        cmd_parts.append('--no-color')

                    # Run command
                    result = subprocess.run(
                        cmd_parts,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )

                    end_time = datetime.utcnow()
                    duration = (end_time - start_time).total_seconds()

                    # Save output if requested
                    if output_dir:
                        output_file = output_path / f"command_{cmd_index:03d}_output.txt"
                        with open(output_file, 'w') as f:
                            f.write(f"Command: {command}\n")
                            f.write(f"Start time: {start_time.isoformat()}\n")
                            f.write(f"End time: {end_time.isoformat()}\n")
                            f.write(f"Duration: {duration:.2f}s\n")
                            f.write(f"Return code: {result.returncode}\n")
                            f.write("\n--- STDOUT ---\n")
                            f.write(result.stdout)
                            f.write("\n--- STDERR ---\n")
                            f.write(result.stderr)

                    return {
                        'command': command,
                        'index': cmd_index,
                        'returncode': result.returncode,
                        'duration': duration,
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'start_time': start_time,
                        'end_time': end_time
                    }

                except subprocess.TimeoutExpired:
                    return {
                        'command': command,
                        'index': cmd_index,
                        'returncode': -1,
                        'duration': timeout,
                        'error': 'Command timeout',
                        'start_time': start_time,
                        'end_time': datetime.utcnow()
                    }
                except Exception as e:
                    return {
                        'command': command,
                        'index': cmd_index,
                        'returncode': -1,
                        'duration': 0,
                        'error': str(e),
                        'start_time': datetime.utcnow(),
                        'end_time': datetime.utcnow()
                    }

            with Progress() as progress:
                task = progress.add_task("[blue]Running commands...", total=len(commands))

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(commands))) as executor:
                    # Submit all commands
                    futures = []
                    for i, command in enumerate(commands):
                        future = executor.submit(run_command, i, command)
                        futures.append(future)

                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        results.append(result)

                        if result['returncode'] != 0:
                            failed_commands.append(result)

                        progress.update(task, advance=1)

                # Sort results by original order
                results.sort(key=lambda x: x['index'])

        else:
            # Sequential execution
            with Progress() as progress:
                task = progress.add_task("[blue]Running commands...", total=len(commands))

                for i, command in enumerate(commands):
                    progress.update(task, description=f"[blue]Running: {command[:50]}...")

                    try:
                        start_time = datetime.utcnow()

                        # Prepare command
                        cmd_parts = command.split()
                        if cmd_parts[0] != 'finlab-cli':
                            cmd_parts = ['finlab-cli'] + cmd_parts

                        # Add common options
                        if ctx.obj.get('verbose'):
                            cmd_parts.extend(['-v'] * ctx.obj['verbose'])
                        if ctx.obj.get('quiet'):
                            cmd_parts.append('-q')
                        if ctx.obj.get('no_color'):
                            cmd_parts.append('--no-color')

                        # Run command
                        result = subprocess.run(
                            cmd_parts,
                            capture_output=True,
                            text=True,
                            timeout=timeout
                        )

                        end_time = datetime.utcnow()
                        duration = (end_time - start_time).total_seconds()

                        cmd_result = {
                            'command': command,
                            'index': i,
                            'returncode': result.returncode,
                            'duration': duration,
                            'stdout': result.stdout,
                            'stderr': result.stderr,
                            'start_time': start_time,
                            'end_time': end_time
                        }

                        results.append(cmd_result)

                        if result.returncode != 0:
                            failed_commands.append(cmd_result)

                            if not continue_on_error:
                                console.print(f"[red]Command failed: {command}[/red]")
                                console.print(f"[red]Error: {result.stderr}[/red]")
                                break

                        # Save output if requested
                        if output_dir:
                            output_file = output_path / f"command_{i:03d}_output.txt"
                            with open(output_file, 'w') as f:
                                f.write(f"Command: {command}\n")
                                f.write(f"Start time: {start_time.isoformat()}\n")
                                f.write(f"End time: {end_time.isoformat()}\n")
                                f.write(f"Duration: {duration:.2f}s\n")
                                f.write(f"Return code: {result.returncode}\n")
                                f.write("\n--- STDOUT ---\n")
                                f.write(result.stdout)
                                f.write("\n--- STDERR ---\n")
                                f.write(result.stderr)

                        progress.update(task, advance=1)

                    except subprocess.TimeoutExpired:
                        cmd_result = {
                            'command': command,
                            'index': i,
                            'returncode': -1,
                            'duration': timeout,
                            'error': 'Command timeout'
                        }
                        results.append(cmd_result)
                        failed_commands.append(cmd_result)

                        if not continue_on_error:
                            console.print(f"[red]Command timeout: {command}[/red]")
                            break

                        progress.update(task, advance=1)

                    except Exception as e:
                        cmd_result = {
                            'command': command,
                            'index': i,
                            'returncode': -1,
                            'duration': 0,
                            'error': str(e)
                        }
                        results.append(cmd_result)
                        failed_commands.append(cmd_result)

                        if not continue_on_error:
                            console.print(f"[red]Command error: {command} - {e}[/red]")
                            break

                        progress.update(task, advance=1)

        # Display results summary
        console.print()
        console.print("[bold]Batch Execution Results[/bold]")

        successful_commands = [r for r in results if r['returncode'] == 0]
        console.print(f"Total commands: {len(commands)}")
        console.print(f"Successful: {len(successful_commands)}")
        console.print(f"Failed: {len(failed_commands)}")

        if results:
            total_duration = sum(r['duration'] for r in results)
            avg_duration = total_duration / len(results)
            console.print(f"Total duration: {format_duration(total_duration)}")
            console.print(f"Average duration: {format_duration(avg_duration)}")

        # Show failed commands
        if failed_commands:
            console.print()
            console.print("[bold red]Failed Commands[/bold red]")

            failed_data = []
            for cmd_result in failed_commands:
                error_msg = cmd_result.get('error', cmd_result.get('stderr', 'Unknown error'))[:50]
                failed_data.append({
                    "Command": cmd_result['command'][:50] + '...' if len(cmd_result['command']) > 50 else cmd_result['command'],
                    "Return Code": cmd_result['returncode'],
                    "Duration": format_duration(cmd_result['duration']),
                    "Error": error_msg + '...' if len(error_msg) > 50 else error_msg
                })

            table = format_table(failed_data, title="Failed Commands")
            console.print(table)

            # Exit with error if any commands failed and not continuing on error
            if not continue_on_error:
                sys.exit(1)

    except Exception as e:
        logger.error(f"Error running batch commands: {e}")
        console.print(f"[red]Error running batch commands: {e}[/red]")
        sys.exit(1)


@batch_group.command()
@click.argument('script_file', type=click.Path(exists=True))
@click.option('--parallel', is_flag=True, help='Run commands in parallel')
@click.option('--continue-on-error', is_flag=True, help='Continue if a command fails')
@click.option('--output-dir', type=click.Path(), help='Directory to save outputs')
@click.pass_context
def script(ctx, script_file: str, parallel: bool, continue_on_error: bool, output_dir: Optional[str]):
    """Run commands from a script file."""
    console = ctx.obj['console']

    try:
        script_path = Path(script_file)
        console.print(f"[blue]Running script: {script_path}[/blue]")

        # Read commands from file
        commands = []
        with open(script_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    commands.append(line)

        if not commands:
            console.print("[yellow]No commands found in script file[/yellow]")
            return

        console.print(f"Found {len(commands)} commands to execute")

        # Run the batch
        ctx.invoke(run,
                  commands=commands,
                  parallel=parallel,
                  continue_on_error=continue_on_error,
                  output_dir=output_dir)

    except Exception as e:
        logger.error(f"Error running script: {e}")
        console.print(f"[red]Error running script: {e}[/red]")
        sys.exit(1)


@batch_group.command()
@click.argument('cron_expression')
@click.argument('command')
@click.option('--name', help='Name for this scheduled job')
@click.option('--output-file', type=click.Path(), help='File to append command outputs')
@click.option('--user', default='$USER', help='User to run the cron job as')
@click.option('--dry-run', is_flag=True, help='Show what would be added without doing it')
@click.pass_context
def schedule(ctx, cron_expression: str, command: str, name: Optional[str],
             output_file: Optional[str], user: str, dry_run: bool):
    """Schedule a command using cron (Linux/WSL)."""
    console = ctx.obj['console']

    try:
        # Validate cron expression (basic validation)
        cron_parts = cron_expression.split()
        if len(cron_parts) != 5:
            raise CliError("Invalid cron expression. Expected format: 'min hour day month weekday'")

        # Prepare the command
        if not command.startswith('finlab-cli'):
            command = f'finlab-cli {command}'

        # Add output redirection if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            command += f' >> {output_path} 2>&1'

        # Create cron entry
        if name:
            cron_comment = f'# {name}'
        else:
            cron_comment = f'# FinLab CLI scheduled task'

        cron_line = f'{cron_expression} {command}'

        console.print(f"[blue]Scheduling command with cron...[/blue]")
        console.print(f"Schedule: {cron_expression}")
        console.print(f"Command: {command}")

        if dry_run:
            console.print()
            console.print("[yellow]DRY RUN - Cron entry that would be added:[/yellow]")
            console.print(cron_comment)
            console.print(cron_line)
            return

        # Check if we're in WSL or Linux
        try:
            result = subprocess.run(['which', 'crontab'], capture_output=True)
            if result.returncode != 0:
                raise CliError("crontab command not found. This feature requires Linux/WSL with cron support.")

            # Get current crontab
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            current_crontab = result.stdout if result.returncode == 0 else ""

            # Add new entry
            new_crontab = current_crontab
            if new_crontab and not new_crontab.endswith('\n'):
                new_crontab += '\n'

            new_crontab += f'{cron_comment}\n{cron_line}\n'

            # Install new crontab
            proc = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
            proc.communicate(input=new_crontab)

            if proc.returncode == 0:
                console.print("[green]✓[/green] Scheduled task added to crontab")
                console.print(f"Next execution: {cron_expression}")
                console.print("Use 'batch list-scheduled' to view all scheduled tasks")
            else:
                raise CliError("Failed to update crontab")

        except FileNotFoundError:
            raise CliError("crontab command not found. This feature requires Linux/WSL with cron support.")

    except CliError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error scheduling command: {e}")
        console.print(f"[red]Error scheduling command: {e}[/red]")
        sys.exit(1)


@batch_group.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.pass_context
def list_scheduled(ctx, output_format: str):
    """List scheduled tasks (cron jobs)."""
    console = ctx.obj['console']

    try:
        # Get current crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)

        if result.returncode != 0:
            console.print("[yellow]No scheduled tasks found (empty crontab)[/yellow]")
            return

        crontab_lines = result.stdout.strip().split('\n')

        # Parse crontab entries
        scheduled_jobs = []
        current_comment = None

        for line in crontab_lines:
            line = line.strip()
            if line.startswith('#'):
                current_comment = line[1:].strip()
            elif line and not line.startswith('#'):
                # Parse cron line
                parts = line.split(None, 5)
                if len(parts) >= 6:
                    cron_schedule = ' '.join(parts[:5])
                    command = parts[5]

                    # Check if it's a finlab-cli command
                    if 'finlab-cli' in command:
                        scheduled_jobs.append({
                            'name': current_comment or 'Unnamed task',
                            'schedule': cron_schedule,
                            'command': command,
                            'full_line': line
                        })

                current_comment = None

        if not scheduled_jobs:
            console.print("[yellow]No FinLab CLI scheduled tasks found[/yellow]")
            return

        if output_format == 'json':
            jobs_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'count': len(scheduled_jobs),
                'jobs': scheduled_jobs
            }
            console.print(json.dumps(jobs_data, indent=2))
        else:
            console.print(f"[bold]Scheduled Tasks ({len(scheduled_jobs)} found)[/bold]")
            console.print()

            jobs_data = []
            for job in scheduled_jobs:
                # Truncate long commands
                cmd_display = job['command']
                if len(cmd_display) > 60:
                    cmd_display = cmd_display[:60] + '...'

                jobs_data.append({
                    'Name': job['name'],
                    'Schedule': job['schedule'],
                    'Command': cmd_display
                })

            table = format_table(jobs_data, title="FinLab CLI Scheduled Tasks")
            console.print(table)

    except FileNotFoundError:
        console.print("[red]crontab command not found. This feature requires Linux/WSL with cron support.[/red]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error listing scheduled tasks: {e}")
        console.print(f"[red]Error listing scheduled tasks: {e}[/red]")
        sys.exit(1)


@batch_group.command()
@click.argument('pattern', required=False)
@click.option('--dry-run', is_flag=True, help='Show what would be removed without doing it')
@click.pass_context
def remove_scheduled(ctx, pattern: Optional[str], dry_run: bool):
    """Remove scheduled tasks matching a pattern."""
    console = ctx.obj['console']

    try:
        # Get current crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)

        if result.returncode != 0:
            console.print("[yellow]No scheduled tasks found (empty crontab)[/yellow]")
            return

        crontab_lines = result.stdout.split('\n')

        # Find matching lines to remove
        new_crontab_lines = []
        removed_lines = []

        i = 0
        while i < len(crontab_lines):
            line = crontab_lines[i].strip()

            # Check if this is a finlab-cli line
            if line and 'finlab-cli' in line:
                # Check if it matches the pattern (if provided)
                matches = True
                if pattern:
                    matches = (pattern.lower() in line.lower() or
                             (i > 0 and pattern.lower() in crontab_lines[i-1].lower()))

                if matches:
                    # Remove this line and any preceding comment
                    if i > 0 and crontab_lines[i-1].strip().startswith('#'):
                        removed_lines.append(crontab_lines[i-1])
                        # Don't add the comment to new_crontab_lines
                        if len(new_crontab_lines) > 0 and new_crontab_lines[-1].strip().startswith('#'):
                            new_crontab_lines.pop()

                    removed_lines.append(line)
                    # Skip this line (don't add to new_crontab_lines)
                else:
                    new_crontab_lines.append(crontab_lines[i])
            else:
                new_crontab_lines.append(crontab_lines[i])

            i += 1

        if not removed_lines:
            if pattern:
                console.print(f"[yellow]No scheduled tasks found matching pattern: {pattern}[/yellow]")
            else:
                console.print("[yellow]No FinLab CLI scheduled tasks found[/yellow]")
            return

        console.print(f"[blue]Found {len([l for l in removed_lines if 'finlab-cli' in l])} tasks to remove[/blue]")

        if dry_run:
            console.print()
            console.print("[yellow]DRY RUN - Lines that would be removed:[/yellow]")
            for line in removed_lines:
                if line.strip():
                    console.print(f"  {line}")
            return

        # Confirm removal
        if not click.confirm(f"Remove {len([l for l in removed_lines if 'finlab-cli' in l])} scheduled tasks?"):
            console.print("Removal cancelled")
            return

        # Update crontab
        new_crontab = '\n'.join(new_crontab_lines)

        proc = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
        proc.communicate(input=new_crontab)

        if proc.returncode == 0:
            console.print(f"[green]✓[/green] Removed {len([l for l in removed_lines if 'finlab-cli' in l])} scheduled tasks")
        else:
            console.print("[red]Error updating crontab[/red]")
            sys.exit(1)

    except FileNotFoundError:
        console.print("[red]crontab command not found. This feature requires Linux/WSL with cron support.[/red]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error removing scheduled tasks: {e}")
        console.print(f"[red]Error removing scheduled tasks: {e}[/red]")
        sys.exit(1)