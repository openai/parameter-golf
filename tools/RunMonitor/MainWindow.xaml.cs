using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using System.Windows.Media;
using Microsoft.Win32;

namespace RunMonitor;

public partial class MainWindow : Window
{
    private static readonly Regex ConfigRegex = new(
        @"^train_batch_tokens:\d+\s+train_seq_len:\d+\s+iterations:(?<iterations>\d+)\s+warmup_steps:(?<warmup>\d+)",
        RegexOptions.Compiled
    );

    private static readonly Regex MaxWallclockRegex = new(
        @"^train_batch_tokens:\d+\s+train_seq_len:\d+\s+iterations:\d+\s+warmup_steps:\d+\s+max_wallclock_seconds:(?<maxwall>[0-9.]+)",
        RegexOptions.Compiled
    );

    private static readonly Regex WarmupRegex = new(
        @"^warmup_step:(?<step>\d+)/(?<total>\d+)$",
        RegexOptions.Compiled
    );

    private static readonly Regex TrainRegex = new(
        @"^step:(?<step>\d+)/(?<total>\d+)\s+train_loss:(?<loss>[0-9.]+)\s+train_time:(?<time>[0-9.]+)ms\s+step_avg:(?<avg>[0-9.]+)ms$",
        RegexOptions.Compiled
    );

    private static readonly Regex ValRegex = new(
        @"^step:(?<step>\d+)/(?<total>\d+)\s+val_loss:(?<loss>[0-9.]+)\s+val_bpb:(?<bpb>[0-9.]+)\s+train_time:(?<time>[0-9.]+)ms\s+step_avg:(?<avg>[0-9.]+)ms$",
        RegexOptions.Compiled
    );

    private static readonly Regex FinalRegex = new(
        @"^final_int8_zlib_roundtrip_exact\s+val_loss:(?<loss>[0-9.]+)\s+val_bpb:(?<bpb>[0-9.]+)$",
        RegexOptions.Compiled
    );

    private static readonly Regex FinalSkippedRegex = new(
        @"^final_int8_zlib_roundtrip skipped by FINAL_ROUNDTRIP_EVAL=0$",
        RegexOptions.Compiled
    );

    private static readonly Regex StopEarlyRegex = new(
        @"^stopping_early:\s+wallclock_cap",
        RegexOptions.Compiled
    );

    private static readonly Regex SerializedInt8Regex = new(
        @"^Serialized model int8\+zlib:\s+(?<bytes>\d+)\s+bytes",
        RegexOptions.Compiled
    );

    private static readonly Regex ArtifactRegex = new(
        @"^Total submission size int8\+zlib:\s+(?<bytes>\d+)\s+bytes$",
        RegexOptions.Compiled
    );

    private static readonly string[] TimestampFormats =
    [
        "ddd MMM d HH:mm:ss yyyy",
        "ddd MMM dd HH:mm:ss yyyy",
    ];

    private readonly DispatcherTimer _timer;
    private readonly string _repoRoot;
    private readonly string _logsDirectory;
    private string? _selectedLogPath;
    private bool _followLatest;
    private const int WarmupPhase = 0;
    private const int InitialValidationPhase = 1;
    private const int TrainingPhase = 2;
    private const int FinalValidationPhase = 3;
    private const int PackagingPhase = 4;
    private const int QuantValidationPhase = 5;
    private const int DonePhase = 6;

    public MainWindow()
    {
        InitializeComponent();
        _repoRoot = FindRepoRoot();
        _logsDirectory = Path.Combine(_repoRoot, "logs");
        _selectedLogPath = ResolveInitialLogPath();
        _followLatest = string.IsNullOrWhiteSpace(_selectedLogPath);
        _timer = new DispatcherTimer
        {
            Interval = TimeSpan.FromSeconds(2),
        };
        _timer.Tick += (_, _) => RefreshSnapshot();
        Loaded += (_, _) =>
        {
            _selectedLogPath ??= FindLatestLogPath();
            RefreshSnapshot();
            _timer.Start();
        };
        Closed += (_, _) => _timer.Stop();
    }

    private void RefreshButton_Click(object sender, RoutedEventArgs e) => RefreshSnapshot();

    private void LatestButton_Click(object sender, RoutedEventArgs e)
    {
        _followLatest = true;
        _selectedLogPath = FindLatestLogPath();
        RefreshSnapshot();
    }

    private void BrowseButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Filter = "Training logs (*.txt)|*.txt|All files (*.*)|*.*",
            InitialDirectory = Directory.Exists(_logsDirectory) ? _logsDirectory : _repoRoot,
            CheckFileExists = true,
        };
        if (dialog.ShowDialog(this) == true)
        {
            _followLatest = false;
            _selectedLogPath = dialog.FileName;
            RefreshSnapshot();
        }
    }

    private void OpenButton_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(_selectedLogPath) || !File.Exists(_selectedLogPath))
        {
            return;
        }

        Process.Start(new ProcessStartInfo
        {
            FileName = _selectedLogPath,
            UseShellExecute = true,
        });
    }

    private void RefreshSnapshot()
    {
        if (_followLatest)
        {
            var latestLogPath = FindLatestLogPath();
            if (!string.IsNullOrWhiteSpace(latestLogPath))
            {
                _selectedLogPath = latestLogPath;
            }
        }

        if (string.IsNullOrWhiteSpace(_selectedLogPath) || !File.Exists(_selectedLogPath))
        {
            _selectedLogPath = FindLatestLogPath();
        }

        if (string.IsNullOrWhiteSpace(_selectedLogPath) || !File.Exists(_selectedLogPath))
        {
            ApplySnapshot(RunSnapshot.Empty("No training log found."));
            return;
        }

        try
        {
            var snapshot = ParseLog(_selectedLogPath);
            ApplySnapshot(snapshot);
        }
        catch (Exception ex)
        {
            ApplySnapshot(RunSnapshot.Empty($"Failed to parse log: {ex.Message}"));
        }
    }

    private void ApplySnapshot(RunSnapshot snapshot)
    {
        RunNameText.Text = snapshot.RunName;
        StageText.Text = snapshot.Stage;
        RunProgressBar.Value = snapshot.ProgressPercent;
        ProgressText.Text = snapshot.ProgressLabel;
        PercentText.Text = $"Run {snapshot.ProgressPercent:0.0}%";
        EtaText.Text = snapshot.EtaLine;
        MetricsText.Text = snapshot.MetricsLine;
        TimingText.Text = snapshot.TimingLine;
        FileText.Text = snapshot.FileLine;
        LastLineText.Text = snapshot.LastLine;
        UpdatedText.Text = snapshot.UpdatedLine;
        ApplyStageTheme(snapshot.Stage);
        ApplyPhaseProgress(snapshot.CurrentPhaseIndex);
    }

    private RunSnapshot ParseLog(string path)
    {
        var lines = File.ReadAllLines(path);
        var fileInfo = new FileInfo(path);

        var iterations = 0;
        var warmupSteps = 0;
        var warmupDone = 0;
        var currentStep = 0;
        var lastStepKind = "Waiting";
        var latestTrainLoss = "";
        var latestValLoss = "";
        var latestValBpb = "";
        var latestTrainTime = "";
        var latestStepAvg = "";
        var finalValLoss = "";
        var finalValBpb = "";
        var artifactBytes = "";
        var lastParsedLine = "No parsed events yet";
        var hasMeasuredStep = false;
        var stopEarlySeen = false;
        var quantArtifactReady = false;
        var finalRoundtripSkipped = false;
        var maxWallclockSeconds = 0.0;
        DateTime? runStartTime = null;

        foreach (var line in lines)
        {
            if (runStartTime is null && TryParseLogTimestamp(line, out var parsedStartTime))
            {
                runStartTime = parsedStartTime;
            }

            var configMatch = ConfigRegex.Match(line);
            if (configMatch.Success)
            {
                iterations = ParseInt(configMatch, "iterations");
                warmupSteps = ParseInt(configMatch, "warmup");
            }

            var wallclockMatch = MaxWallclockRegex.Match(line);
            if (wallclockMatch.Success)
            {
                maxWallclockSeconds = ParseDouble(wallclockMatch, "maxwall");
            }

            var warmupMatch = WarmupRegex.Match(line);
            if (warmupMatch.Success)
            {
                warmupDone = ParseInt(warmupMatch, "step");
                warmupSteps = Math.Max(warmupSteps, ParseInt(warmupMatch, "total"));
                lastStepKind = warmupDone < warmupSteps ? "Warmup" : "Warmup complete";
                lastParsedLine = line;
            }

            var trainMatch = TrainRegex.Match(line);
            if (trainMatch.Success)
            {
                currentStep = ParseInt(trainMatch, "step");
                iterations = Math.Max(iterations, ParseInt(trainMatch, "total"));
                latestTrainLoss = trainMatch.Groups["loss"].Value;
                latestTrainTime = trainMatch.Groups["time"].Value;
                latestStepAvg = trainMatch.Groups["avg"].Value;
                lastStepKind = "Training";
                lastParsedLine = line;
                hasMeasuredStep = true;
            }

            var valMatch = ValRegex.Match(line);
            if (valMatch.Success)
            {
                currentStep = ParseInt(valMatch, "step");
                iterations = Math.Max(iterations, ParseInt(valMatch, "total"));
                latestValLoss = valMatch.Groups["loss"].Value;
                latestValBpb = valMatch.Groups["bpb"].Value;
                latestTrainTime = valMatch.Groups["time"].Value;
                latestStepAvg = valMatch.Groups["avg"].Value;
                lastStepKind = currentStep >= iterations && iterations > 0 ? "Final validation" : "Validation";
                lastParsedLine = line;
                hasMeasuredStep = true;
            }

            var finalMatch = FinalRegex.Match(line);
            if (finalMatch.Success)
            {
                finalValLoss = finalMatch.Groups["loss"].Value;
                finalValBpb = finalMatch.Groups["bpb"].Value;
                lastStepKind = "Finished";
                lastParsedLine = line;
                hasMeasuredStep = true;
            }

            if (FinalSkippedRegex.IsMatch(line))
            {
                finalRoundtripSkipped = true;
                lastStepKind = "Finished";
                lastParsedLine = line;
            }

            if (StopEarlyRegex.IsMatch(line))
            {
                stopEarlySeen = true;
                lastParsedLine = line;
            }

            var serializedInt8Match = SerializedInt8Regex.Match(line);
            if (serializedInt8Match.Success)
            {
                artifactBytes = serializedInt8Match.Groups["bytes"].Value;
                quantArtifactReady = true;
                lastParsedLine = line;
            }

            var artifactMatch = ArtifactRegex.Match(line);
            if (artifactMatch.Success)
            {
                artifactBytes = artifactMatch.Groups["bytes"].Value;
                quantArtifactReady = true;
                lastParsedLine = line;
            }
        }

        if (iterations <= 0)
        {
            iterations = 1;
        }

        var logAge = DateTime.Now - fileInfo.LastWriteTime;
        var effectiveCurrentStep = currentStep;
        if (
            currentStep > 0 &&
            latestStepAvg.Length > 0 &&
            double.TryParse(latestStepAvg, NumberStyles.Float, CultureInfo.InvariantCulture, out var liveStepAvgMs) &&
            liveStepAvgMs > 0.0
        )
        {
            effectiveCurrentStep = Math.Min(
                iterations,
                currentStep + (int)Math.Floor(Math.Max(logAge.TotalMilliseconds, 0.0) / liveStepAvgMs)
            );
        }

        var waitingOnInitialValidation = !hasMeasuredStep && warmupSteps > 0 && warmupDone >= warmupSteps && currentStep == 0;
        var capExceededWithoutFinal = maxWallclockSeconds > 0.0 &&
            double.TryParse(latestTrainTime, NumberStyles.Float, CultureInfo.InvariantCulture, out var latestTrainTimeMsSnapshot) &&
            latestTrainTimeMsSnapshot >= maxWallclockSeconds * 1000.0 &&
            finalValBpb.Length == 0;
        var proxyComplete = finalRoundtripSkipped;
        var progressPercent = CalculateProgressPercent(
            finalValBpb,
            proxyComplete,
            waitingOnInitialValidation,
            capExceededWithoutFinal,
            stopEarlySeen,
            quantArtifactReady,
            warmupDone,
            warmupSteps,
            currentStep,
            effectiveCurrentStep,
            iterations,
            latestStepAvg,
            latestTrainTime,
            maxWallclockSeconds,
            runStartTime ?? fileInfo.CreationTime,
            fileInfo.LastWriteTime,
            logAge
        );
        if (proxyComplete)
        {
            lastStepKind = "Finished";
        }
        else if (waitingOnInitialValidation)
        {
            lastStepKind = "Initial validation";
        }
        else if (quantArtifactReady)
        {
            lastStepKind = "Quantized validation";
        }
        else if (stopEarlySeen)
        {
            lastStepKind = "Packaging";
        }
        else if (capExceededWithoutFinal)
        {
            lastStepKind = "Final validation";
        }

        string progressLabel;
        if (finalValBpb.Length > 0)
        {
            progressLabel = "Finished";
        }
        else if (proxyComplete)
        {
            progressLabel = $"Finished at step {currentStep} | final roundtrip eval skipped";
        }
        else if (waitingOnInitialValidation)
        {
            progressLabel = $"Warmup {warmupDone}/{Math.Max(warmupSteps, warmupDone)} | waiting on step 0 validation";
        }
        else if (quantArtifactReady)
        {
            progressLabel = $"Cap reached at step {currentStep} | validating quantized artifact";
        }
        else if (stopEarlySeen)
        {
            progressLabel = $"Cap reached at step {currentStep} | packaging artifact";
        }
        else if (capExceededWithoutFinal)
        {
            progressLabel = $"Cap reached at step {currentStep} | waiting on final validation";
        }
        else if (warmupDone > 0 && currentStep == 0)
        {
            progressLabel = $"Warmup {warmupDone}/{Math.Max(warmupSteps, warmupDone)}";
        }
        else if (warmupDone > 0 && currentStep > 0)
        {
            progressLabel = effectiveCurrentStep > currentStep
                ? $"Warmup {warmupDone}/{Math.Max(warmupSteps, warmupDone)} | est. step {effectiveCurrentStep}/{iterations} (last log {currentStep})"
                : $"Warmup {warmupDone}/{Math.Max(warmupSteps, warmupDone)} | step {currentStep}/{iterations}";
        }
        else
        {
            progressLabel = $"Step {currentStep}/{iterations}";
        }

        var metrics = "No metrics yet";
        if (finalValBpb.Length > 0)
        {
            metrics = $"final val_loss {finalValLoss} | final val_bpb {finalValBpb}";
        }
        else if (proxyComplete)
        {
            var proxyMetricParts = new List<string>();
            if (latestValLoss.Length > 0)
            {
                proxyMetricParts.Add($"proxy val_loss {latestValLoss}");
            }
            if (latestValBpb.Length > 0)
            {
                proxyMetricParts.Add($"proxy val_bpb {latestValBpb}");
            }
            if (artifactBytes.Length > 0)
            {
                proxyMetricParts.Add($"artifact_bytes {artifactBytes}");
            }
            metrics = proxyMetricParts.Count > 0
                ? string.Join(" | ", proxyMetricParts)
                : "Proxy run complete; int8 artifact exported and final roundtrip eval skipped";
        }
        else if (waitingOnInitialValidation)
        {
            metrics = "Waiting for the first full validation pass after warmup";
        }
        else if (quantArtifactReady)
        {
            metrics = "Quantized artifact ready; running the post-quantization validation pass";
        }
        else if (stopEarlySeen)
        {
            metrics = "Training has stopped; exporting and reloading the quantized artifact";
        }
        else if (capExceededWithoutFinal)
        {
            metrics = "Reached the training wallclock cap; waiting for the final validation pass";
        }
        else
        {
            var metricParts = new List<string>();
            if (latestTrainLoss.Length > 0)
            {
                metricParts.Add($"train_loss {latestTrainLoss}");
            }
            if (latestValLoss.Length > 0)
            {
                metricParts.Add($"val_loss {latestValLoss}");
            }
            if (latestValBpb.Length > 0)
            {
                metricParts.Add($"val_bpb {latestValBpb}");
            }
            if (metricParts.Count > 0)
            {
                metrics = string.Join(" | ", metricParts);
            }
        }

        var timingParts = new List<string>();
        if (latestTrainTime.Length > 0)
        {
            timingParts.Add($"train_time_ms {latestTrainTime}");
        }
        if (latestStepAvg.Length > 0)
        {
            timingParts.Add($"step_avg_ms {latestStepAvg}");
        }
        if (artifactBytes.Length > 0)
        {
            timingParts.Add($"artifact_bytes {artifactBytes}");
        }
        if (timingParts.Count == 0)
        {
            timingParts.Add("Waiting for first measured step");
        }
        if (waitingOnInitialValidation)
        {
            timingParts.Add("full validation runs before step 1");
        }
        if (stopEarlySeen)
        {
            timingParts.Add("training stopped at wallclock cap");
        }
        if (quantArtifactReady && !proxyComplete)
        {
            timingParts.Add("post-quant validation in progress");
        }
        if (proxyComplete)
        {
            timingParts.Add("proxy run complete");
        }
        if (capExceededWithoutFinal && !proxyComplete)
        {
            timingParts.Add("training cap hit; final eval in progress");
        }
        if (logAge.TotalSeconds >= 5)
        {
            timingParts.Add($"last_log_update {FormatAge(logAge)} ago");
        }

        var etaLine = BuildEtaLine(
            finalValBpb,
            proxyComplete,
            waitingOnInitialValidation,
            capExceededWithoutFinal,
            stopEarlySeen,
            quantArtifactReady,
            currentStep,
            effectiveCurrentStep,
            iterations,
            warmupDone,
            warmupSteps,
            latestStepAvg,
            latestTrainTime,
            maxWallclockSeconds,
            runStartTime ?? fileInfo.CreationTime,
            fileInfo.LastWriteTime,
            logAge
        );
        var currentPhaseIndex = DetermineCurrentPhaseIndex(
            finalValBpb,
            proxyComplete,
            waitingOnInitialValidation,
            capExceededWithoutFinal,
            stopEarlySeen,
            quantArtifactReady,
            warmupDone,
            warmupSteps,
            currentStep,
            hasMeasuredStep
        );

        return new RunSnapshot(
            RunName: Path.GetFileNameWithoutExtension(path),
            Stage: lastStepKind,
            CurrentPhaseIndex: currentPhaseIndex,
            ProgressPercent: Math.Clamp(progressPercent, 0.0, 100.0),
            ProgressLabel: progressLabel,
            EtaLine: etaLine,
            MetricsLine: metrics,
            TimingLine: string.Join(" | ", timingParts),
            FileLine: path,
            LastLine: lastParsedLine,
            UpdatedLine: $"Log updated {fileInfo.LastWriteTime:yyyy-MM-dd HH:mm:ss} | {FormatAge(logAge)} ago | polling every 2s"
        );
    }

    private static int ParseInt(Match match, string name)
    {
        return int.TryParse(match.Groups[name].Value, out var value) ? value : 0;
    }

    private static double ParseDouble(Match match, string name)
    {
        return double.TryParse(match.Groups[name].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
            ? value
            : 0.0;
    }

    private static bool TryParseLogTimestamp(string line, out DateTime timestamp)
    {
        return DateTime.TryParseExact(
            line.Trim(),
            TimestampFormats,
            CultureInfo.InvariantCulture,
            DateTimeStyles.AllowWhiteSpaces | DateTimeStyles.AssumeLocal,
            out timestamp
        );
    }

    private static string FormatAge(TimeSpan age)
    {
        if (age.TotalHours >= 1)
        {
            return $"{Math.Floor(age.TotalHours)}h {age.Minutes}m";
        }
        if (age.TotalMinutes >= 1)
        {
            return $"{Math.Floor(age.TotalMinutes)}m {age.Seconds}s";
        }
        return $"{Math.Max(age.Seconds, 0)}s";
    }

    private static string FormatClockTime(DateTime time)
    {
        return time.ToString("h:mm tt", CultureInfo.InvariantCulture);
    }

    private static string FormatDuration(TimeSpan span)
    {
        if (span.TotalHours >= 1)
        {
            return $"{Math.Floor(span.TotalHours)}h {span.Minutes}m";
        }
        if (span.TotalMinutes >= 1)
        {
            return $"{Math.Floor(span.TotalMinutes)}m {span.Seconds}s";
        }
        return $"{Math.Max(span.Seconds, 0)}s";
    }

    private static string BuildEtaLine(
        string finalValBpb,
        bool proxyComplete,
        bool waitingOnInitialValidation,
        bool capExceededWithoutFinal,
        bool stopEarlySeen,
        bool quantArtifactReady,
        int currentStep,
        int effectiveCurrentStep,
        int iterations,
        int warmupDone,
        int warmupSteps,
        string latestStepAvg,
        string latestTrainTime,
        double maxWallclockSeconds,
        DateTime runStartTime,
        DateTime lastLogWriteTime,
        TimeSpan logAge)
    {
        if (finalValBpb.Length > 0)
        {
            return "Complete";
        }

        if (proxyComplete)
        {
            return "Complete (roundtrip skipped)";
        }

        if (capExceededWithoutFinal)
        {
            var remainingWrapUp = EstimateWrapUpRemaining(
                stopEarlySeen,
                quantArtifactReady,
                warmupDone,
                warmupSteps,
                latestStepAvg,
                latestTrainTime,
                runStartTime,
                lastLogWriteTime,
                logAge
            );
            if (remainingWrapUp.HasValue)
            {
                var stageLabel = quantArtifactReady
                    ? "quant eval"
                    : stopEarlySeen
                        ? "packaging + quant eval"
                        : "final + quant eval";
                if (remainingWrapUp.Value > TimeSpan.Zero)
                {
                    var finishTime = DateTime.Now + remainingWrapUp.Value;
                    return $"{FormatClockTime(finishTime)} (~{FormatDuration(remainingWrapUp.Value)} left, {stageLabel})";
                }
                return $"Over expected time by {FormatDuration(remainingWrapUp.Value.Duration())} ({stageLabel})";
            }
            return "Wrapping up final eval";
        }

        var hasStepAvg = double.TryParse(latestStepAvg, NumberStyles.Float, CultureInfo.InvariantCulture, out var stepAvgMs);
        var hasTrainTime = double.TryParse(latestTrainTime, NumberStyles.Float, CultureInfo.InvariantCulture, out var trainTimeMs);

        TimeSpan? remainingBySteps = null;
        if (hasStepAvg && effectiveCurrentStep > 0 && iterations > effectiveCurrentStep)
        {
            var ageMs = Math.Max(logAge.TotalMilliseconds, 0.0);
            var ageIntoCurrentStepMs = stepAvgMs > 0.0 ? ageMs % stepAvgMs : 0.0;
            var remainingStepFractionMs = ageIntoCurrentStepMs > 0.0 ? stepAvgMs - ageIntoCurrentStepMs : 0.0;
            var remainingFullSteps = Math.Max(iterations - Math.Max(effectiveCurrentStep, currentStep + 1), 0);
            remainingBySteps = TimeSpan.FromMilliseconds(remainingStepFractionMs + stepAvgMs * remainingFullSteps);
        }

        TimeSpan? remainingByCap = null;
        if (maxWallclockSeconds > 0.0 && hasTrainTime)
        {
            var remainingCapMs = maxWallclockSeconds * 1000.0 - (trainTimeMs + Math.Max(logAge.TotalMilliseconds, 0.0));
            if (remainingCapMs > 0.0)
            {
                remainingByCap = TimeSpan.FromMilliseconds(remainingCapMs);
            }
        }

        TimeSpan? chosenRemaining = null;
        var capBound = false;
        if (remainingBySteps.HasValue && remainingByCap.HasValue)
        {
            if (remainingByCap.Value <= remainingBySteps.Value)
            {
                chosenRemaining = remainingByCap;
                capBound = true;
            }
            else
            {
                chosenRemaining = remainingBySteps;
            }
        }
        else if (remainingBySteps.HasValue)
        {
            chosenRemaining = remainingBySteps;
        }
        else if (remainingByCap.HasValue)
        {
            chosenRemaining = remainingByCap;
            capBound = true;
        }

        if (chosenRemaining.HasValue)
        {
            var finishTime = DateTime.Now + chosenRemaining.Value;
            var qualifier = capBound ? " cap-bound" : "";
            if (capBound && hasStepAvg && stepAvgMs > 0.0 && remainingByCap.HasValue)
            {
                var projectedStopStep = Math.Min(
                    iterations,
                    effectiveCurrentStep + (int)Math.Floor(remainingByCap.Value.TotalMilliseconds / stepAvgMs)
                );
                return $"{FormatClockTime(finishTime)} ({FormatDuration(chosenRemaining.Value)} left{qualifier}, ~step {projectedStopStep})";
            }
            return $"{FormatClockTime(finishTime)} ({FormatDuration(chosenRemaining.Value)} left{qualifier})";
        }

        if (waitingOnInitialValidation && maxWallclockSeconds > 0.0)
        {
            var nominalCapTime = runStartTime + TimeSpan.FromSeconds(maxWallclockSeconds);
            var remaining = nominalCapTime - DateTime.Now;
            if (remaining > TimeSpan.Zero)
            {
                return $"{FormatClockTime(nominalCapTime)} (~{FormatDuration(remaining)} left, pending first step)";
            }
        }

        if (waitingOnInitialValidation)
        {
            return "Waiting for first measured step";
        }

        return "ETA unavailable";
    }

    private static TimeSpan? EstimateWrapUpRemaining(
        bool stopEarlySeen,
        bool quantArtifactReady,
        int warmupDone,
        int warmupSteps,
        string latestStepAvg,
        string latestTrainTime,
        DateTime runStartTime,
        DateTime lastLogWriteTime,
        TimeSpan logAge)
    {
        var validationPassEstimate = EstimateValidationPassDuration(
            stopEarlySeen,
            quantArtifactReady,
            warmupDone,
            warmupSteps,
            latestStepAvg,
            latestTrainTime,
            runStartTime,
            lastLogWriteTime
        );
        if (!validationPassEstimate.HasValue)
        {
            return null;
        }

        var packagingEstimate = TimeSpan.FromMinutes(2);
        TimeSpan stageEstimate;
        if (quantArtifactReady)
        {
            stageEstimate = validationPassEstimate.Value;
        }
        else if (stopEarlySeen)
        {
            stageEstimate = packagingEstimate + validationPassEstimate.Value;
        }
        else
        {
            stageEstimate = validationPassEstimate.Value + packagingEstimate + validationPassEstimate.Value;
        }

        var remaining = stageEstimate - logAge;
        return remaining;
    }

    private static TimeSpan? EstimateValidationPassDuration(
        bool stopEarlySeen,
        bool quantArtifactReady,
        int warmupDone,
        int warmupSteps,
        string latestStepAvg,
        string latestTrainTime,
        DateTime runStartTime,
        DateTime lastLogWriteTime)
    {
        var hasStepAvg = double.TryParse(latestStepAvg, NumberStyles.Float, CultureInfo.InvariantCulture, out var stepAvgMs);
        var hasTrainTime = double.TryParse(latestTrainTime, NumberStyles.Float, CultureInfo.InvariantCulture, out var trainTimeMs);

        double estimatedMs = 0.0;
        if (hasTrainTime)
        {
            var elapsedSinceStartMs = Math.Max((lastLogWriteTime - runStartTime).TotalMilliseconds, 0.0);
            var warmupCount = warmupDone > 0 ? warmupDone : warmupSteps;
            var warmupEstimateMs = hasStepAvg && warmupCount > 0 ? stepAvgMs * warmupCount : 0.0;
            var packagingEstimateMs = TimeSpan.FromMinutes(2).TotalMilliseconds;
            var nonTrainingEvalMs = elapsedSinceStartMs - trainTimeMs - warmupEstimateMs;
            if (quantArtifactReady)
            {
                nonTrainingEvalMs -= packagingEstimateMs;
            }

            var completedValidationPasses = (stopEarlySeen || quantArtifactReady) ? 2.0 : 1.0;
            estimatedMs = nonTrainingEvalMs / completedValidationPasses;
        }

        if (estimatedMs <= 0.0 && hasStepAvg)
        {
            estimatedMs = stepAvgMs * 100.0;
        }

        if (estimatedMs <= 0.0)
        {
            return null;
        }

        estimatedMs = Math.Clamp(estimatedMs, 120000.0, 7200000.0);
        return TimeSpan.FromMilliseconds(estimatedMs);
    }

    private static double CalculateProgressPercent(
        string finalValBpb,
        bool proxyComplete,
        bool waitingOnInitialValidation,
        bool capExceededWithoutFinal,
        bool stopEarlySeen,
        bool quantArtifactReady,
        int warmupDone,
        int warmupSteps,
        int currentStep,
        int effectiveCurrentStep,
        int iterations,
        string latestStepAvg,
        string latestTrainTime,
        double maxWallclockSeconds,
        DateTime runStartTime,
        DateTime lastLogWriteTime,
        TimeSpan logAge)
    {
        const double warmupSpan = 8.0;
        const double initialValidationSpan = 4.0;
        const double trainingStart = warmupSpan + initialValidationSpan;
        const double finalValidationStart = 96.0;
        const double trainingSpan = finalValidationStart - trainingStart;
        const double wrapUpSpan = 4.0;

        if (finalValBpb.Length > 0 || proxyComplete)
        {
            return 100.0;
        }

        if (capExceededWithoutFinal)
        {
            var wrapUpProgress = EstimateWrapUpProgressFraction(
                stopEarlySeen,
                quantArtifactReady,
                warmupDone,
                warmupSteps,
                latestStepAvg,
                latestTrainTime,
                runStartTime,
                lastLogWriteTime,
                logAge
            );
            if (wrapUpProgress.HasValue)
            {
                return Math.Clamp(finalValidationStart + wrapUpSpan * wrapUpProgress.Value, finalValidationStart, 99.8);
            }
            return 99.0;
        }

        if (warmupSteps > 0 && warmupDone < warmupSteps && currentStep == 0)
        {
            return Math.Clamp(warmupSpan * warmupDone / Math.Max(warmupSteps, 1), 0.0, warmupSpan);
        }

        if (waitingOnInitialValidation)
        {
            return warmupSpan + initialValidationSpan * 0.5;
        }

        if (
            maxWallclockSeconds > 0.0 &&
            double.TryParse(latestTrainTime, NumberStyles.Float, CultureInfo.InvariantCulture, out var trainTimeMs)
        )
        {
            var capMs = maxWallclockSeconds * 1000.0;
            var liveTrainTimeMs = trainTimeMs + Math.Max(logAge.TotalMilliseconds, 0.0);
            var trainingProgress = capMs > 0.0 ? Math.Clamp(liveTrainTimeMs / capMs, 0.0, 1.0) : 0.0;
            return Math.Clamp(trainingStart + trainingSpan * trainingProgress, 0.0, finalValidationStart);
        }

        if (iterations > 0)
        {
            var stepProgress = Math.Clamp((double)Math.Max(effectiveCurrentStep, currentStep) / iterations, 0.0, 1.0);
            return Math.Clamp(trainingStart + trainingSpan * stepProgress, 0.0, finalValidationStart);
        }

        if (warmupSteps > 0)
        {
            return Math.Clamp(warmupSpan * warmupDone / Math.Max(warmupSteps, 1), 0.0, warmupSpan);
        }

        return 0.0;
    }

    private static double? EstimateWrapUpProgressFraction(
        bool stopEarlySeen,
        bool quantArtifactReady,
        int warmupDone,
        int warmupSteps,
        string latestStepAvg,
        string latestTrainTime,
        DateTime runStartTime,
        DateTime lastLogWriteTime,
        TimeSpan logAge)
    {
        var validationPassEstimate = EstimateValidationPassDuration(
            stopEarlySeen,
            quantArtifactReady,
            warmupDone,
            warmupSteps,
            latestStepAvg,
            latestTrainTime,
            runStartTime,
            lastLogWriteTime
        );
        if (!validationPassEstimate.HasValue)
        {
            return null;
        }

        var packagingEstimate = TimeSpan.FromMinutes(2);
        var totalDuration = validationPassEstimate.Value + packagingEstimate + validationPassEstimate.Value;

        TimeSpan completedDuration;
        if (quantArtifactReady)
        {
            completedDuration = validationPassEstimate.Value + packagingEstimate + logAge;
        }
        else if (stopEarlySeen)
        {
            completedDuration = validationPassEstimate.Value + logAge;
        }
        else
        {
            completedDuration = logAge;
        }

        var fraction = completedDuration.TotalMilliseconds / Math.Max(totalDuration.TotalMilliseconds, 1.0);
        return Math.Clamp(fraction, 0.0, 0.95);
    }

    private static int DetermineCurrentPhaseIndex(
        string finalValBpb,
        bool proxyComplete,
        bool waitingOnInitialValidation,
        bool capExceededWithoutFinal,
        bool stopEarlySeen,
        bool quantArtifactReady,
        int warmupDone,
        int warmupSteps,
        int currentStep,
        bool hasMeasuredStep)
    {
        if (finalValBpb.Length > 0 || proxyComplete)
        {
            return DonePhase;
        }

        if (quantArtifactReady)
        {
            return QuantValidationPhase;
        }

        if (stopEarlySeen)
        {
            return PackagingPhase;
        }

        if (capExceededWithoutFinal)
        {
            return FinalValidationPhase;
        }

        if (waitingOnInitialValidation)
        {
            return InitialValidationPhase;
        }

        if (warmupSteps > 0 && warmupDone < warmupSteps)
        {
            return WarmupPhase;
        }

        if (currentStep > 0 || hasMeasuredStep)
        {
            return TrainingPhase;
        }

        return WarmupPhase;
    }

    private void ApplyPhaseProgress(int currentPhaseIndex)
    {
        var phaseBorders = new[]
        {
            PhaseWarmupBorder,
            PhaseInitialValBorder,
            PhaseTrainingBorder,
            PhaseFinalValBorder,
            PhasePackagingBorder,
            PhaseQuantValBorder,
            PhaseDoneBorder,
        };
        var phaseTexts = new[]
        {
            PhaseWarmupText,
            PhaseInitialValText,
            PhaseTrainingText,
            PhaseFinalValText,
            PhasePackagingText,
            PhaseQuantValText,
            PhaseDoneText,
        };

        for (var i = 0; i < phaseBorders.Length; i++)
        {
            var isCompleted = currentPhaseIndex > i;
            var isCurrent = currentPhaseIndex == i;

            string background;
            string border;
            string foreground;

            if (isCurrent)
            {
                (background, border, foreground) = i switch
                {
                    WarmupPhase => ("#1F2A46", "#4C63A8", "#C7D4FF"),
                    InitialValidationPhase or FinalValidationPhase or QuantValidationPhase => ("#3B2B12", "#916E31", "#FFD98B"),
                    TrainingPhase or DonePhase => ("#183424", "#2F6E4A", "#8AF0B1"),
                    PackagingPhase => ("#17303C", "#2D6C82", "#84D8F5"),
                    _ => ("#182634", "#314458", "#D6E1EC"),
                };
            }
            else if (isCompleted)
            {
                background = "#171D23";
                border = "#3B4652";
                foreground = "#8E9BA8";
            }
            else
            {
                background = "#10161D";
                border = "#24313D";
                foreground = "#5F7181";
            }

            phaseBorders[i].Background = BrushFromHex(background);
            phaseBorders[i].BorderBrush = BrushFromHex(border);
            phaseTexts[i].Foreground = BrushFromHex(foreground);
            phaseTexts[i].FontWeight = isCurrent ? FontWeights.SemiBold : FontWeights.Normal;
        }
    }

    private void ApplyStageTheme(string stage)
    {
        var normalized = stage.ToLowerInvariant();
        var (background, border, foreground) = normalized switch
        {
            var s when s.Contains("finished") => ("#183424", "#2F6E4A", "#8AF0B1"),
            var s when s.Contains("validation") => ("#2A2114", "#7A6138", "#F5C46D"),
            var s when s.Contains("training") => ("#182D24", "#2D7759", "#75E0AE"),
            var s when s.Contains("warmup") => ("#1F2230", "#48527A", "#AEBBFF"),
            _ => ("#182634", "#314458", "#D6E1EC"),
        };

        StageBadgeBorder.Background = BrushFromHex(background);
        StageBadgeBorder.BorderBrush = BrushFromHex(border);
        StageText.Foreground = BrushFromHex(foreground);
    }

    private static Brush BrushFromHex(string hex)
    {
        return (SolidColorBrush)new BrushConverter().ConvertFromString(hex)!;
    }

    private string? ResolveInitialLogPath()
    {
        var args = Environment.GetCommandLineArgs();
        foreach (var arg in args.Skip(1))
        {
            if (File.Exists(arg))
            {
                return Path.GetFullPath(arg);
            }
        }
        return null;
    }

    private string? FindLatestLogPath()
    {
        if (!Directory.Exists(_logsDirectory))
        {
            return null;
        }

        return Directory.EnumerateFiles(_logsDirectory, "*.txt", SearchOption.TopDirectoryOnly)
            .Where(path =>
                !path.EndsWith(".stdout.txt", StringComparison.OrdinalIgnoreCase) &&
                !path.EndsWith(".stderr.txt", StringComparison.OrdinalIgnoreCase) &&
                !path.EndsWith(".controller.txt", StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(File.GetLastWriteTimeUtc)
            .FirstOrDefault();
    }

    private static string FindRepoRoot()
    {
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "train_gpt.py")))
            {
                return current.FullName;
            }
            current = current.Parent;
        }
        return Directory.GetCurrentDirectory();
    }

    private sealed record RunSnapshot(
        string RunName,
        string Stage,
        int CurrentPhaseIndex,
        double ProgressPercent,
        string ProgressLabel,
        string EtaLine,
        string MetricsLine,
        string TimingLine,
        string FileLine,
        string LastLine,
        string UpdatedLine)
    {
        public static RunSnapshot Empty(string reason) => new(
            RunName: "No run selected",
            Stage: "Idle",
            CurrentPhaseIndex: -1,
            ProgressPercent: 0.0,
            ProgressLabel: "No progress yet",
            EtaLine: "ETA unavailable",
            MetricsLine: reason,
            TimingLine: "Waiting for a readable log",
            FileLine: "No log selected",
            LastLine: reason,
            UpdatedLine: $"Updated {DateTime.Now:yyyy-MM-dd HH:mm:ss} | polling every 2s"
        );
    }
}
